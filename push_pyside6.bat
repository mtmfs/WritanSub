@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo === WritanSub PySide6 分支推送脚本 ===
echo.

:: 检查是否已配置 git 用户
git config user.name >nul 2>&1
if errorlevel 1 (
    echo [!] 请先配置 git 用户信息：
    echo     git config --global user.name "你的用户名"
    echo     git config --global user.email "你的邮箱"
    echo.
    pause
    exit /b 1
)

:: 切换或创建 pyside6 分支
echo [1/4] 切换到 pyside6 分支...
git checkout pyside6 2>nul || git checkout -b pyside6

:: 添加修改的文件
echo [2/4] 添加修改的文件...
git add pyproject.toml writansub/gui/

:: 提交
echo [3/4] 提交更改...
git commit -m "feat: migrate GUI from Tkinter to PySide6"

:: 推送
echo [4/4] 推送到远程...
git push -u origin pyside6

echo.
echo === 完成! ===
pause
