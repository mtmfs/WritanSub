@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

where uv >nul 2>&1 (
    uv run python -m writansub.cli %*
    goto :end
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m writansub.cli %*
    goto :end
)

python -m writansub.cli %*

:end
pause
