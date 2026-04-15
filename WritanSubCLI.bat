@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set HF_HUB_OFFLINE=1
set HF_HUB_CACHE=G:\本地部署模型\faster-whisper
.venv\Scripts\python.exe -m writansub.cli %*
pause
