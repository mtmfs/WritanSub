@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

where uv >nul 2>&1 (
    uv run python -m writansub %*
    exit /b
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m writansub %*
    exit /b
)

python -m writansub %*
