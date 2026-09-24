#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kD           = 256;
constexpr std::int32_t kQHeads      = 24;
constexpr std::int32_t kKVHeads     = 4;
constexpr std::int32_t kGroup       = 64;
constexpr std::int32_t kGroups      = kD / kGroup;
constexpr std::int32_t kQueryGroup  = kQHeads / kKVHeads;
constexpr std::int32_t kPackedD     = kD / 2;
constexpr float kAttentionScale     = 0.0625f;
constexpr std::uint16_t kOutCanary  = 0x7fc1u; // quiet NaN when interpreted as BF16

// RK8V4 is an input-storage profile here: the oracle decodes exactly the independently generated
// K8/V4 cache, then computes ideal attention in FP64. This envelope covers Q8 tensor-core staging,
// FP16 V staging, BF16 output storage, and the final inverse rotation, not V4 quantization itself.
constexpr ReductionCriterion kRk8v4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 6.0e-3,
    /*gross_relative_to_max_reference*/ 3.0e-2,
};

enum class Mapping { Identity, Fragmented };

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    Mapping mapping;
    bool graph_replay;
    std::uint32_t seed;
};

const char* mapping_name(Mapping mapping) {
    return mapping == Mapping::Identity ? "identity" : "fragmented";
}

std::size_t q_index(std::int32_t d, std::int32_t head, std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
}

std::size_t logical_index(std::int32_t leading, std::int32_t capacity, std::int32_t d,
                          std::int32_t position, std::int32_t head) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(leading) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(capacity) * static_cast<std::size_t>(head));
}

std::size_t physical_index(std::int32_t leading, std::int32_t d, std::int32_t page_offset,
                           std::int32_t head, std::int32_t physical_page) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(leading) *
               (static_cast<std::size_t>(page_offset) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kKVHeads) *
                         static_cast<std::size_t>(physical_page)));
}

std::uint16_t float_to_half_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t fraction = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (fraction == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        const std::uint32_t significand = fraction | 0x800000u;
        const int shift                 = 14 - half_exponent;
        std::uint32_t rounded           = significand >> shift;
        const std::uint32_t remainder   = significand & ((std::uint32_t{1} << shift) - 1u);
        const std::uint32_t halfway     = std::uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0)) { ++rounded; }
        return static_cast<std::uint16_t>(sign | rounded);
    }

    std::uint32_t rounded         = fraction >> 13;
    const std::uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0)) {
        ++rounded;
        if (rounded == 0x400u) {
            rounded = 0;
            ++half_exponent;
            if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(half_exponent) << 10) |
                                      static_cast<std::uint16_t>(rounded));
}

float half_bits_to_float(std::uint16_t bits) {
    const bool negative       = (bits & 0x8000u) != 0;
    const std::uint16_t e     = static_cast<std::uint16_t>((bits >> 10) & 0x1fu);
    const std::uint16_t m     = static_cast<std::uint16_t>(bits & 0x03ffu);
    float value               = 0.0f;
    if (e == 0) {
        value = std::ldexp(static_cast<float>(m), -24);
    } else if (e == 31) {
        value = m == 0 ? std::numeric_limits<float>::infinity()
                       : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0f + static_cast<float>(m) / 1024.0f,
                           static_cast<int>(e) - 15);
    }
    return negative ? -value : value;
}

int round_to_nearest_even(float value) {
    const double lower    = std::floor(static_cast<double>(value));
    const double fraction = static_cast<double>(value) - lower;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(std::fabs(lower), 2.0) == 1.0)) {
        return static_cast<int>(lower + 1.0);
    }
    return static_cast<int>(lower);
}

// Test-owned scalar Walsh-Hadamard transform. It intentionally does not include or call the CUDA
// shuffle implementation. H64 is orthonormal and self-inverse.
template <typename T>
void hadamard64(std::array<T, kGroup>& values) {
    for (std::int32_t stride = 1; stride < kGroup; stride *= 2) {
        for (std::int32_t block = 0; block < kGroup; block += 2 * stride) {
            for (std::int32_t i = 0; i < stride; ++i) {
                const T a = values[static_cast<std::size_t>(block + i)];
                const T b = values[static_cast<std::size_t>(block + stride + i)];
                values[static_cast<std::size_t>(block + i)]          = a + b;
                values[static_cast<std::size_t>(block + stride + i)] = a - b;
            }
        }
    }
    for (T& value : values) value *= static_cast<T>(0.125);
}

std::int8_t quantize_symmetric(float value, float scale, int limit) {
    if (scale == 0.0f) return 0;
    const int code = std::clamp(round_to_nearest_even(value / scale), -limit, limit);
    return static_cast<std::int8_t>(code);
}

std::uint8_t pack_i4(std::int8_t low, std::int8_t high) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(low) & 0x0fu) |
                                     ((static_cast<unsigned>(high) & 0x0fu) << 4));
}

std::int8_t unpack_i4(std::uint8_t packed, bool high) {
    const unsigned nibble = high ? packed >> 4 : packed & 0x0fu;
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

struct HostRkCache {
    std::int32_t capacity = 0;
    std::vector<std::int8_t> k_codes;
    std::vector<std::uint8_t> v_packed;
    std::vector<std::uint16_t> k_scales;
    std::vector<std::uint16_t> v_scales;
    std::vector<float> decoded_k;
    std::vector<float> decoded_v;
};

void decode_cache(HostRkCache& cache) {
    const std::size_t decoded_count = static_cast<std::size_t>(kD) * cache.capacity * kKVHeads;
    cache.decoded_k.assign(decoded_count, 0.0f);
    cache.decoded_v.assign(decoded_count, 0.0f);
    for (std::int32_t head = 0; head < kKVHeads; ++head) {
        for (std::int32_t position = 0; position < cache.capacity; ++position) {
            for (std::int32_t group = 0; group < kGroups; ++group) {
                const std::size_t scale_at = logical_index(kGroups, cache.capacity, group,
                                                           position, head);
                const float k_scale = half_bits_to_float(cache.k_scales[scale_at]);
                const float v_scale = half_bits_to_float(cache.v_scales[scale_at]);
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    const std::int32_t d = group * kGroup + i;
                    const std::size_t dst = logical_index(kD, cache.capacity, d, position, head);
                    cache.decoded_k[dst] = static_cast<float>(cache.k_codes[dst]) * k_scale;
                    const std::size_t packed_at =
                        logical_index(kPackedD, cache.capacity, d / 2, position, head);
                    cache.decoded_v[dst] =
                        static_cast<float>(unpack_i4(cache.v_packed[packed_at], (d & 1) != 0)) *
                        v_scale;
                }
            }
        }
    }
}

HostRkCache make_cache(std::int32_t capacity, std::uint32_t seed) {
    const std::size_t source_count = static_cast<std::size_t>(kD) * capacity * kKVHeads;
    std::vector<float> source_k(source_count);
    std::vector<float> source_v(source_count);
    fill_uniform(source_k, seed, -0.30f, 0.30f);
    fill_uniform(source_v, seed + 1u, -1.0f, 1.0f);
    round_to_bf16(source_k);
    round_to_bf16(source_v);

    // Exercise the scale==0 codec branch in a visible row without making the whole fixture
    // degenerate. All other groups remain seeded, mixed-sign activation data.
    for (std::int32_t d = 0; d < kGroup; ++d) {
        source_k[logical_index(kD, capacity, d, 0, 0)] = 0.0f;
        source_v[logical_index(kD, capacity, d, 0, 0)] = 0.0f;
    }

    HostRkCache cache;
    cache.capacity = capacity;
    cache.k_codes.resize(source_count);
    cache.v_packed.resize(static_cast<std::size_t>(kPackedD) * capacity * kKVHeads);
    cache.k_scales.resize(static_cast<std::size_t>(kGroups) * capacity * kKVHeads);
    cache.v_scales.resize(static_cast<std::size_t>(kGroups) * capacity * kKVHeads);

    for (std::int32_t head = 0; head < kKVHeads; ++head) {
        for (std::int32_t position = 0; position < capacity; ++position) {
            for (std::int32_t group = 0; group < kGroups; ++group) {
                std::array<float, kGroup> k_values{};
                std::array<float, kGroup> v_values{};
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    const std::int32_t d = group * kGroup + i;
                    const std::size_t at = logical_index(kD, capacity, d, position, head);
                    k_values[static_cast<std::size_t>(i)] = source_k[at];
                    v_values[static_cast<std::size_t>(i)] = source_v[at];
                }
                hadamard64(k_values);
                hadamard64(v_values);

                float k_absmax = 0.0f;
                float v_absmax = 0.0f;
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    k_absmax = std::max(k_absmax, std::fabs(k_values[static_cast<std::size_t>(i)]));
                    v_absmax = std::max(v_absmax, std::fabs(v_values[static_cast<std::size_t>(i)]));
                }
                const std::uint16_t k_scale_bits =
                    float_to_half_bits(k_absmax > 0.0f ? k_absmax / 127.0f : 0.0f);
                const std::uint16_t v_scale_bits =
                    float_to_half_bits(v_absmax > 0.0f ? v_absmax / 7.0f : 0.0f);
                const float k_scale = half_bits_to_float(k_scale_bits);
                const float v_scale = half_bits_to_float(v_scale_bits);
                const std::size_t scale_at =
                    logical_index(kGroups, capacity, group, position, head);
                cache.k_scales[scale_at] = k_scale_bits;
                cache.v_scales[scale_at] = v_scale_bits;

                for (std::int32_t i = 0; i < kGroup; ++i) {
                    const std::int32_t d = group * kGroup + i;
                    const std::size_t at = logical_index(kD, capacity, d, position, head);
                    cache.k_codes[at] =
                        quantize_symmetric(k_values[static_cast<std::size_t>(i)], k_scale, 127);
                }
                for (std::int32_t i = 0; i < kGroup; i += 2) {
                    const std::int8_t low =
                        quantize_symmetric(v_values[static_cast<std::size_t>(i)], v_scale, 7);
                    const std::int8_t high =
                        quantize_symmetric(v_values[static_cast<std::size_t>(i + 1)], v_scale, 7);
                    const std::uint8_t packed = pack_i4(low, high);
                    if (unpack_i4(packed, false) != low || unpack_i4(packed, true) != high) {
                        throw std::runtime_error("independent V4 codec round-trip failed");
                    }
                    const std::int32_t packed_d = (group * kGroup + i) / 2;
                    cache.v_packed[logical_index(kPackedD, capacity, packed_d, position, head)] =
                        packed;
                }
            }
        }
    }
    decode_cache(cache);
    return cache;
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostRkCache& cache,
                                    std::span<const std::int32_t> positions) {
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> rotated_q(q.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < kQHeads; ++head) {
            for (std::int32_t group = 0; group < kGroups; ++group) {
                std::array<double, kGroup> values{};
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    values[static_cast<std::size_t>(i)] =
                        static_cast<double>(q[q_index(group * kGroup + i, head, token)]);
                }
                hadamard64(values);
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    rotated_q[q_index(group * kGroup + i, head, token)] =
                        values[static_cast<std::size_t>(i)];
                }
            }
        }
    }

    std::vector<double> output(q.size());
    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> weights(scores.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < kQHeads; ++q_head) {
            const std::int32_t kv_head = q_head / kQueryGroup;
            double maximum             = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kD; ++d) {
                    dot += rotated_q[q_index(d, q_head, token)] *
                           static_cast<double>(cache.decoded_k[logical_index(
                               kD, cache.capacity, d, position, kv_head)]);
                }
                scores[static_cast<std::size_t>(position)] =
                    dot * static_cast<double>(kAttentionScale);
                maximum = std::max(maximum, scores[static_cast<std::size_t>(position)]);
            }
            double denominator = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                const double weight =
                    std::exp(scores[static_cast<std::size_t>(position)] - maximum);
                weights[static_cast<std::size_t>(position)] = weight;
                denominator += weight;
            }

            for (std::int32_t group = 0; group < kGroups; ++group) {
                std::array<double, kGroup> rotated_output{};
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    const std::int32_t d = group * kGroup + i;
                    double value         = 0.0;
                    for (std::int32_t position = 0; position < visible; ++position) {
                        value += (weights[static_cast<std::size_t>(position)] / denominator) *
                                 static_cast<double>(cache.decoded_v[logical_index(
                                     kD, cache.capacity, d, position, kv_head)]);
                    }
                    rotated_output[static_cast<std::size_t>(i)] = value;
                }
                hadamard64(rotated_output);
                for (std::int32_t i = 0; i < kGroup; ++i) {
                    output[q_index(group * kGroup + i, q_head, token)] =
                        rotated_output[static_cast<std::size_t>(i)];
                }
            }
        }
    }
    return output;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, Mapping mapping) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    for (std::int32_t page = 0; page < logical_pages; ++page) {
        table[static_cast<std::size_t>(page)] =
            mapping == Mapping::Identity ? page : 2 * page + 1;
    }
    return table;
}

template <typename T>
std::vector<T> scatter_pages(const std::vector<T>& logical, std::int32_t leading,
                             std::int32_t capacity, std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages, T sentinel) {
    std::vector<T> physical(static_cast<std::size_t>(leading) * kPagedKVPageSize * kKVHeads *
                                physical_pages,
                            sentinel);
    for (std::int32_t head = 0; head < kKVHeads; ++head) {
        for (std::int32_t position = 0; position < capacity; ++position) {
            const std::int32_t physical_page =
                block_table[static_cast<std::size_t>(position / kPagedKVPageSize)];
            const std::int32_t page_offset = position % kPagedKVPageSize;
            for (std::int32_t d = 0; d < leading; ++d) {
                physical[physical_index(leading, d, page_offset, head, physical_page)] =
                    logical[logical_index(leading, capacity, d, position, head)];
            }
        }
    }
    return physical;
}

struct PhysicalCache {
    std::int32_t logical_pages  = 0;
    std::int32_t physical_pages = 0;
    std::vector<std::int32_t> block_table;
    std::vector<std::int8_t> k;
    std::vector<std::uint8_t> v;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

PhysicalCache make_physical_cache(const HostRkCache& logical, Mapping mapping) {
    PhysicalCache result;
    result.logical_pages  = logical.capacity / kPagedKVPageSize;
    result.physical_pages = mapping == Mapping::Identity ? result.logical_pages
                                                          : 2 * result.logical_pages + 1;
    result.block_table = make_block_table(result.logical_pages, mapping);
    result.k = scatter_pages(logical.k_codes, kD, logical.capacity, result.block_table,
                             result.physical_pages, static_cast<std::int8_t>(0x55));
    result.v = scatter_pages(logical.v_packed, kPackedD, logical.capacity, result.block_table,
                             result.physical_pages, static_cast<std::uint8_t>(0xcdu));
    result.k_scale = scatter_pages(logical.k_scales, kGroups, logical.capacity,
                                   result.block_table, result.physical_pages,
                                   static_cast<std::uint16_t>(0x7bffu));
    result.v_scale = scatter_pages(logical.v_scales, kGroups, logical.capacity,
                                   result.block_table, result.physical_pages,
                                   static_cast<std::uint16_t>(0x7bffu));
    return result;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

std::vector<std::uint16_t> bf16_bits(std::span<const float> values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) result[i] = f32_to_bf16(values[i]);
    return result;
}

int verify_cache_unchanged(const std::string& label, const PhysicalCache& expected,
                           const GuardedDeviceBuffer& k, const GuardedDeviceBuffer& v,
                           const GuardedDeviceBuffer& ks, const GuardedDeviceBuffer& vs,
                           const GuardedDeviceBuffer& table) {
    int failures = 0;
    failures += verify_exact((label + " k codes").c_str(),
                             copy_from_guarded<std::int8_t>(k, expected.k.size()), expected.k);
    failures += verify_exact((label + " packed v").c_str(),
                             copy_from_guarded<std::uint8_t>(v, expected.v.size()), expected.v);
    failures += verify_exact((label + " k scales").c_str(),
                             copy_from_guarded<std::uint16_t>(ks, expected.k_scale.size()),
                             expected.k_scale);
    failures += verify_exact((label + " v scales").c_str(),
                             copy_from_guarded<std::uint16_t>(vs, expected.v_scale.size()),
                             expected.v_scale);
    failures += verify_exact((label + " block table").c_str(),
                             copy_from_guarded<std::int32_t>(table,
                                                             expected.block_table.size()),
                             expected.block_table);
    failures += k.verify_guards(label + " k guard");
    failures += v.verify_guards(label + " v guard");
    failures += ks.verify_guards(label + " k-scale guard");
    failures += vs.verify_guards(label + " v-scale guard");
    failures += table.verify_guards(label + " table guard");
    return failures;
}

int run_case(const AttentionCase& test_case) {
    const std::int32_t total = test_case.base + test_case.tokens;
    const std::int32_t capacity =
        ((total + kPagedKVPageSize - 1) / kPagedKVPageSize) * kPagedKVPageSize;
    const HostRkCache logical = make_cache(capacity, test_case.seed + 10u);
    const PhysicalCache physical = make_physical_cache(logical, test_case.mapping);

    const std::size_t q_count =
        static_cast<std::size_t>(kD) * kQHeads * test_case.tokens;
    std::vector<float> q(q_count);
    fill_uniform(q, test_case.seed, -0.25f, 0.25f);
    round_to_bf16(q);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const std::vector<double> reference = ideal_attention(q, logical, positions);
    const std::vector<std::uint16_t> q_storage = bf16_bits(q);
    const std::vector<std::uint16_t> output_canary(q_count, kOutCanary);

    GuardedDeviceBuffer dq(q_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dk(physical.k.size() * sizeof(std::int8_t));
    GuardedDeviceBuffer dv(physical.v.size() * sizeof(std::uint8_t));
    GuardedDeviceBuffer dks(physical.k_scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dvs(physical.v_scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dtable(physical.block_table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_count * sizeof(std::uint16_t));
    dq.copy_from_host(q_storage.data(), dq.bytes());
    dpositions.copy_from_host(positions.data(), dpositions.bytes());
    dk.copy_from_host(physical.k.data(), dk.bytes());
    dv.copy_from_host(physical.v.data(), dv.bytes());
    dks.copy_from_host(physical.k_scale.data(), dks.bytes());
    dvs.copy_from_host(physical.v_scale.data(), dvs.bytes());
    dtable.copy_from_host(physical.block_table.data(), dtable.bytes());
    dout.copy_from_host(output_canary.data(), dout.bytes());

    Tensor tq(dq.data(), DType::BF16, {kD, kQHeads, test_case.tokens});
    Tensor tp(dpositions.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kD, kQHeads, test_case.tokens});
    PagedKVLayerView cache{
        .k_pages = Tensor(dk.data(), DType::I8,
                          {kD, kPagedKVPageSize, kKVHeads, physical.physical_pages}),
        .v_pages = Tensor(dv.data(), DType::U8,
                          {kPackedD, kPagedKVPageSize, kKVHeads, physical.physical_pages}),
        .k_scale_pages = Tensor(dks.data(), DType::FP16,
                                {kGroups, kPagedKVPageSize, kKVHeads, physical.physical_pages}),
        .v_scale_pages = Tensor(dvs.data(), DType::FP16,
                                {kGroups, kPagedKVPageSize, kKVHeads, physical.physical_pages}),
        .block_table = Tensor(dtable.data(), DType::I32, {physical.logical_pages}),
        .head_dim = kD,
        .num_kv_heads = kKVHeads,
        .dtype = DType::I8,
        .quant_group = kGroup,
        .packed_v = true,
        .rotate_k = true,
        .rotate_v = true,
    };
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             static_cast<std::uint32_t>(capacity)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, DType::I8, envelope, 1, test_case.tokens, test_case.tokens);
    if (workspace_bytes != 0) {
        std::cerr << "RK8V4 prompt fixture unexpectedly selected a workspace route\n";
        return 1;
    }
    GuardedDeviceBuffer workspace_storage(256);
    WorkspaceArena workspace(DeviceSpan{workspace_storage.data(), workspace_storage.bytes()});

    const std::string label =
        "gqa_attention_cached rk8v4 T=" + std::to_string(test_case.tokens) +
        " base=" + std::to_string(test_case.base) + " mapping=" +
        mapping_name(test_case.mapping) + (test_case.graph_replay ? " graph" : " eager");
    int failures = 0;
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create RK8V4 GQA stream");

    if (!test_case.graph_replay) {
        ops::gqa_attention_cached(tq, tp, kAttentionScale, cache, envelope, workspace, tout,
                                  stream);
        cuda_synchronize(stream);
        failures += verify_reduction(label, from_device_bf16(dout.data(), q_count), reference,
                                     kRk8v4Criterion);
    } else {
        cudaGraph_t graph          = nullptr;
        cudaGraphExec_t executable = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "begin RK8V4 GQA capture");
        ops::gqa_attention_cached(tq, tp, kAttentionScale, cache, envelope, workspace, tout,
                                  stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end RK8V4 GQA capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                   "instantiate RK8V4 GQA graph");
        for (int replay = 0; replay < 2; ++replay) {
            dout.copy_from_host(output_canary.data(), dout.bytes());
            cuda_check(cudaGraphLaunch(executable, stream), "launch RK8V4 GQA graph");
            cuda_synchronize(stream);
            failures += verify_reduction(label + " replay=" + std::to_string(replay),
                                         from_device_bf16(dout.data(), q_count), reference,
                                         kRk8v4Criterion);
        }
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);

    failures += verify_exact((label + " q unchanged").c_str(),
                             copy_from_guarded<std::uint16_t>(dq, q_storage.size()), q_storage);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             copy_from_guarded<std::int32_t>(dpositions, positions.size()),
                             positions);
    failures += dq.verify_guards(label + " q guard");
    failures += dpositions.verify_guards(label + " positions guard");
    failures += dout.verify_guards(label + " output guard");
    failures += workspace_storage.verify_guards(label + " workspace guard");
    failures += verify_cache_unchanged(label + " cache unchanged", physical, dk, dv, dks, dvs,
                                       dtable);
    if (workspace.used() != 0 || workspace.peak_used() != 0) {
        std::cerr << label << ": prompt route used workspace\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // T=17 is the first public prompt route; base=63 crosses a logical page boundary; T=65
    // crosses the prompt kernel's 64-query CTA boundary. Both direct and fragmented page maps are
    // represented, and the widest case proves capture plus repeat replay of the public A3 call.
    constexpr AttentionCase cases[] = {
        {17, 0, Mapping::Identity, false, 4101u},
        {17, 63, Mapping::Fragmented, false, 4201u},
        {65, 64, Mapping::Fragmented, true, 4301u},
    };
    int failures = 0;
    for (const AttentionCase& test_case : cases) failures += run_case(test_case);
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention RK8V4 cached-prompt correctness\n";
    return failures == 0 ? 0 : 1;
}
