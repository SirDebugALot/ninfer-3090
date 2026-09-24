# GitHub 공개 방법 - 처음 사용하는 분용

이 작업 트리는 이미 공개용 브랜치 `rtx3090-faster-prefill`에 있습니다.
저장소는 `Don-Chad/ninfer-3090`이고, 공개 제목은 **NInfer RTX 3090 -
Faster Prefill**을 권장합니다. 성능 구현은 Round 11 기준이며, 이후의
UTF-8 복구는 별도의 안정성 수정입니다.

공개 README의 성능 수치는 공식 v0.6.1 Windows 바이너리와 Round 11의
32K INT8 동일 조건 A/B/B/A 결과인 **785.03 -> 1037.43 tok/s (+32.15%)**를
사용합니다. 공식 바이너리는 RK8V4 benchmark를 지원하지 않으므로 RK8V4
수치를 원본과의 직접 비교처럼 표시하지 않습니다.

## 무엇을 올리나

GitHub에는 이 저장소 루트의 소스와 문서를 올립니다. 즉 현재 위치의
`CMakeLists.txt`, `src/`, `include/`, `apps/`, `bench/`, `tests/`, `scripts/`,
`docs/`, `.github/`, `README.md`, `LICENSE` 등을 커밋합니다.

올리지 않는 것:

- `.ninfer`, GGUF, 체크포인트 등 모델 파일
- `build*`, `vcpkg_installed`, CUDA 설치 폴더
- `ninfer-prefill-lab`의 라운드별 패키지와 원시 로그
- `results-*`, GPU telemetry, 요청 JSONL과 개인 경로가 든 로그
- 탈락한 Round 12-23 후보 소스/실행파일
- 이 작업 공간의 상세 연구 일지 `docs/rtx-3090-prefill-optimization.md`

이 항목들은 `.gitignore`로 대부분 차단됩니다. 바이너리는 소스 커밋에
넣지 않고 GitHub Actions artifact 또는 GitHub Release asset으로 올립니다.

## 1. 변경 내용을 확인하고 커밋하기

PowerShell에서 저장소 폴더로 이동한 뒤 다음을 실행합니다.

```powershell
git status --short
git branch --show-current
git diff --check
```

브랜치 이름이 `rtx3090-faster-prefill`인지 확인합니다. 그 다음 GitHub에
공개할 파일만 스테이징합니다. `git add -A`는 상세 로컬 연구 파일까지
실수로 넣을 수 있으므로 사용하지 않는 편이 안전합니다.

```powershell
git add -u
git add .github README.md PUBLISHING-KO.md RELEASE_NOTES_PREFILL.md
git add docs/rtx-3090-faster-prefill.md
git add scripts/package-prefill-windows.ps1 scripts/run-qwen38-prefill-c6-64k.bat
git add bench/ops/gqa_attention_bench.cu
git add src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cublas.cu
git add src/ops/linear_add/q5/q5_linear_add_cublas.cu
git add src/ops/linear_swiglu/q4/q4_linear_swiglu_full_cuda126.cu
git add tests/ops/test_gqa_attention_rk8v4.cpp
git status --short
git diff --cached --stat
git commit -m "perf(sm86): accelerate Qwen prefill on RTX 3090"
```

마지막 두 명령에서 모델·로그·라운드 패키지가 보이면 커밋하지 말고
`git restore --staged <파일>`로 제외합니다.

## 2. 브랜치를 GitHub에 올리기

```powershell
git push -u origin rtx3090-faster-prefill
```

GitHub 저장소 화면의 브랜치 선택 메뉴에서 `rtx3090-faster-prefill`을
선택하면 “3090용 최적화 브랜치”임이 바로 표시됩니다. 기존 `main`을
유지하려면 Compare & pull request를 눌러 PR을 만들고 제목을 다음처럼
지정합니다.

```text
perf(sm86): faster Qwen3.8 prefill on RTX 3090
```

이 브랜치를 저장소의 대표판으로 쓸 경우 GitHub의 Settings > Branches에서
default branch를 이 브랜치로 바꿀 수도 있습니다. 초보자에게는 우선 PR로
내용을 확인한 다음 main에 병합하는 방식을 권장합니다.

## 3. Actions에서 Windows ZIP 만들기

워크플로 파일은 `.github/workflows/windows-prefill-release.yml`입니다.
GitHub-hosted Windows PC에서 CUDA 13.2와 12.6, vcpkg를 설치하고 SM86
Release를 컴파일합니다. 실제 GPU 실행 테스트는 하지 않으며, 컴파일과
패키징만 수행합니다.

워크플로가 기본 브랜치에 들어간 후 GitHub에서 Actions > Windows RTX
3090 prefill release > Run workflow를 누릅니다. 완료되면 실행 화면 아래
Artifacts에서 Windows ZIP을 내려받을 수 있습니다. 첫 실행은 두 CUDA
툴킷과 FFmpeg를 빌드하므로 오래 걸릴 수 있습니다.

`Jimver/cuda-toolkit`은 GitHub 인증 공식 Action이 아닌 외부 Action입니다.
공개 전에 워크플로의 버전과 코드를 검토하십시오. 더 엄격한 공급망
정책이 필요하면 두 CUDA가 미리 설치된 본인 Windows PC를 self-hosted
runner로 연결하는 방식이 좋습니다.

## 4. GitHub Release에 자동으로 올리기

컴파일이 확인된 뒤 태그를 만들고 push합니다.

```powershell
git tag -a v0.6.1-rtx3090-prefill1 -m "NInfer RTX 3090 faster prefill"
git push origin v0.6.1-rtx3090-prefill1
```

태그가 push되면 같은 Action이 다시 빌드하고, ZIP과 SHA-256 파일을
GitHub Releases에 자동으로 첨부합니다. Releases 화면의 설명은
`RELEASE_NOTES_PREFILL.md`에서 가져옵니다. 모델 파일은 별도로 받습니다.

수동으로 올리고 싶다면 GitHub의 Releases > Draft a new release에서 같은
태그를 고른 다음 Actions가 만든 ZIP과 `SHA256SUMS-*.txt`를 끌어다 놓고
Publish release를 누르면 됩니다.

## 주의

Actions 빌드는 “컴파일에 성공했다”는 증거입니다. GitHub-hosted runner에는
RTX 3090이 없으므로 실제 모델 로드와 성능 검증은 로컬 RTX 3090에서 별도로
해야 합니다. 공개 Release에는 로컬에서 검증한 커밋의 태그만 사용하십시오.
