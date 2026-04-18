@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

where uv >nul 2>&1
if %errorlevel%==0 (
    echo [WritanSub-CLI] launching via uv
    uv run python -m writansub.cli %*
    goto :end
)

if exist ".venv\Scripts\python.exe" (
    echo [WritanSub-CLI] launching via .venv\Scripts\python.exe
    .venv\Scripts\python.exe -m writansub.cli %*
    goto :end
)

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WritanSub-CLI] ERROR: no Python runtime found
    echo   tried: uv, .venv\Scripts\python.exe, python on PATH
    echo   install uv from https://docs.astral.sh/uv/ or create .venv and install deps
    pause
    exit /b 1
)

echo [WritanSub-CLI] launching via system python on PATH
python -m writansub.cli %*

:end
pause
