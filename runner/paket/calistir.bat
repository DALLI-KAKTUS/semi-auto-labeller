@echo off
rem ENG yerel calistirici (Windows): bu dosyaya cift tiklayin. Ilk acilista uv ve gerekli paketleri kurar.
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv kuruluyor ^(bir kez; https://docs.astral.sh/uv/^)...
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  where uv >nul 2>nul || goto :fail
)

if not exist .venv\Scripts\python.exe goto :install
if not exist .venv\.kurulum-tamam goto :install
goto :run

:install
echo Python ortami hazirlaniyor ^(ilk acilista birkac dakika surebilir^)...
uv venv --python 3.11 --allow-existing .venv || goto :fail
uv pip install --python .venv --torch-backend auto -r requirements-runner.txt || goto :fail
type nul > .venv\.kurulum-tamam

:run
.venv\Scripts\python.exe -W ignore -m runner %*
pause
exit /b

:fail
echo Kurulum basarisiz ^(internet baglantisini kontrol edip yeniden deneyin^).
pause
exit /b 1
