@echo off
rem Yerel calistirici (Windows): scripts\run_runner.bat --server https://... --token engr_... --model models\lite_v8.json=C:\yol\lite_net.pt
cd /d "%~dp0.."
.venv\Scripts\python.exe -W ignore -m runner %*
