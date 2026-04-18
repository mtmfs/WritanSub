@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

where uv >nul 2>&1
if %errorlevel%==0 (
    echo [WritanSub] launching via uv
    uv run python -m writansub %*
    goto :end
)

if exist ".venv\Scripts\python.exe" (
    echo [WritanSub] launching via .venv\Scripts\python.exe
    .venv\Scripts\python.exe -m writansub %*
    goto :end
)

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WritanSub] ERROR: no Python runtime found
    echo   tried: uv ^(not on PATH^), .venv\Scripts\python.exe ^(not present^), python ^(not on PATH^)
    echo   install uv from https://docs.astral.sh/uv/ or create .venv and install deps
    pause
    exit /b 1
)

echo [WritanSub] launching via system python on PATH
python -m writansub %*

:end
if errorlevel 1 (
    echo.
    echo [WritanSub] exited with errorlevel %errorlevel%
    pause
)
