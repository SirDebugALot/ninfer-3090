@echo off
setlocal EnableExtensions
title NInfer RTX 3090 - Qwen3.8-27B faster prefill - C6 64K

set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"
set "PORT=8005"
set "LOG_DIR=%ROOT%logs"

if not "%~1"=="" set "MODEL=%~1"
if not exist "%SERVER%" (
  echo ERROR: Missing "%SERVER%".
  exit /b 1
)
if not exist "%MODEL%" (
  echo ERROR: Missing model "%MODEL%".
  echo Run download-qwen38.bat or drag qwen3_8_27b.ninfer onto this file.
  exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%T in ('powershell.exe -NoProfile -Command "[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')"') do set "RUN_STAMP=%%T"
set "REQUEST_LOG=%LOG_DIR%\requests-%RUN_STAMP%.jsonl"

echo API: http://127.0.0.1:%PORT%/v1
echo Profile: shared 64K, concurrency 6, RK8V4, chunk 2048, MTP3, CUDA Graphs
echo Request log: "%REQUEST_LOG%"
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port %PORT% --device 0 --model-id qwen3.8-27b --max-context 65536 --kv-capacity 65536 --max-concurrency 6 --max-pending-requests 32 --pending-timeout-ms 900000 --prefill-chunk 2048 --kv-dtype rk8v4 --spec mtp --draft-tokens 3 --lm-head-draft --request-log-jsonl "%REQUEST_LOG%"

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo NInfer exited with error code %EXIT_CODE%.
pause
endlocal & exit /b %EXIT_CODE%
