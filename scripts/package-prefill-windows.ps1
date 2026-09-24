param(
    [string]$BuildRoot = 'build-prefill-windows',
    [string]$ReleaseLabel = 'v0.6.1-rtx3090-prefill1',
    [Parameter(Mandatory = $true)][string]$CudaRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $BuildRoot))
$distRoot = Join-Path $repoRoot 'dist'
$productName = "ninfer-rtx3090-faster-prefill-windows-x64-$ReleaseLabel"
$productRoot = Join-Path $distRoot $productName
$archivePath = Join-Path $distRoot ($productName + '.zip')
$checksumPath = Join-Path $distRoot ('SHA256SUMS-' + $ReleaseLabel + '.txt')

New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $distRoot).Path
if ([IO.Path]::GetFullPath((Split-Path -Parent $productRoot)) -ne $resolvedDist) {
    throw 'Refusing to package outside dist.'
}
if (Test-Path -LiteralPath $productRoot) { Remove-Item -LiteralPath $productRoot -Recurse -Force }
if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
New-Item -ItemType Directory -Path $productRoot | Out-Null

$products = @(
    @{ Source = 'apps/ninfer.exe'; Destination = 'ninfer.exe' },
    @{ Source = 'apps/ninfer-serve.exe'; Destination = 'ninfer-serve.exe' },
    @{ Source = 'bench/ninfer_bench.exe'; Destination = 'ninfer_bench.exe' }
)
foreach ($product in $products) {
    $source = Join-Path $buildPath $product.Source
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $productRoot $product.Destination)
}

$vcpkgBinCandidates = @(
    (Join-Path $buildPath 'vcpkg_installed/x64-windows/bin'),
    (Join-Path $repoRoot 'vcpkg_installed/x64-windows/bin')
)
$vcpkgBin = $vcpkgBinCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
    Select-Object -First 1
if (!$vcpkgBin) {
    throw "Missing vcpkg runtime directory. Checked: $($vcpkgBinCandidates -join ', ')"
}

$dllSources = @()
$dllSources += Get-ChildItem -LiteralPath $vcpkgBin -Filter '*.dll' -File
$dllSources += Get-ChildItem -LiteralPath (Join-Path $CudaRoot 'bin') -Filter 'cublas*.dll' -File
foreach ($dll in $dllSources | Sort-Object Name -Unique) {
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $productRoot $dll.Name)
}

Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') -Destination $productRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'VERSION') -Destination $productRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/rtx-3090-windows.md') -Destination (Join-Path $productRoot 'README.md')
Copy-Item -LiteralPath (Join-Path $repoRoot 'RELEASE_NOTES_PREFILL.md') -Destination $productRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/download-qwen38.bat') -Destination $productRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/run-qwen38-prefill-c6-64k.bat') -Destination $productRoot
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/run-qwen38-c6-96k-rk8v4.bat') -Destination $productRoot

$innerHashes = Get-ChildItem -LiteralPath $productRoot -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
}
$innerHashes | Set-Content -LiteralPath (Join-Path $productRoot 'SHA256SUMS.txt') -Encoding ascii
Compress-Archive -LiteralPath $productRoot -DestinationPath $archivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
"$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $archivePath)" |
    Set-Content -LiteralPath $checksumPath -Encoding ascii
Get-Item -LiteralPath $archivePath,$checksumPath | Select-Object Name,Length
