@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_8_27b.ninfer"
set "PART=%MODEL%.part"
set "MODEL_REVISION=18dfc887423fa5aabf3cb56fac41490e462b3fab"
set "MODEL_SHA256=eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if exist "%MODEL%" (
  call :verify "%MODEL%"
  if not errorlevel 1 goto model_ready
  echo ERROR: Existing model does not match the required container-v2 artifact.
  echo Move or delete "%MODEL%" and run this file again.
  exit /b 1
)

echo Downloading the pinned Qwen3.8-27B NInfer container-v2 model...
echo Revision: %MODEL_REVISION%
curl.exe -L -C - --fail --output "%PART%" "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/%MODEL_REVISION%/qwen3_8_27b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)

echo Verifying SHA-256...
call :verify "%PART%"
if errorlevel 1 (
  echo ERROR: Downloaded model failed SHA-256 verification.
  echo Expected: %MODEL_SHA256%
  exit /b 1
)
move /Y "%PART%" "%MODEL%" >nul

:model_ready
echo Model ready: %MODEL%
exit /b 0

:verify
set "ACTUAL_SHA256="
for /f "usebackq delims=" %%H in (`certutil.exe -hashfile "%~1" SHA256 ^| findstr.exe /r /x /i "[0-9a-f][0-9a-f]*"`) do set "ACTUAL_SHA256=%%H"
if /I "%ACTUAL_SHA256%"=="%MODEL_SHA256%" exit /b 0
exit /b 1
