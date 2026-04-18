#Requires -Version 5.1
<#
Build the lightweight source bundle at build/dist/WritanSub-light/.

Layout produced:
  WritanSub-light/
    WritanSub.exe          <- same Rust launcher as full bundle
    WritanSubCLI.exe
    app/writansub/         <- git archive HEAD
    runtime/               <- empty; install.bat fills it from vendor/python-embed.zip
    vendor/
      python-embed.zip     <- official Windows embeddable, ~10 MB (bundled so install is offline for Python itself)
      writansub_native-<ver>-cp312-cp312-win_amd64.whl
    requirements.lock.txt
    install.bat            <- 5-stage installer (extract Python -> patch _pth -> ensure uv -> uv pip install -> native wheel)
    README.txt

Runtime structure after install.bat finishes is byte-compatible with the full bundle,
so both flavours use the same WritanSub.exe.
#>

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$SkipNative,
    [switch]$SkipLauncher
)

$ErrorActionPreference = 'Stop'

$RepoRoot  = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$BuildRoot = Join-Path $RepoRoot 'build'
$DistRoot  = Join-Path $BuildRoot 'dist'
$LightDist = Join-Path $DistRoot 'WritanSub-light'
$VendorDir = Join-Path $LightDist 'vendor'
$AppDir    = Join-Path $LightDist 'app'
$Cache     = Join-Path $BuildRoot 'cache'
$PythonVer = '3.12.10'

$RustupBin = Join-Path $env:USERPROFILE '.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin'
if ((Test-Path $RustupBin) -and ($env:PATH -notlike "*$RustupBin*")) {
    $env:PATH = "$RustupBin;$env:PATH"
}

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" -ForegroundColor DarkGray }

# ---------- C1. Version check ----------
Step "Checking version alignment"
$version = & (Join-Path $PSScriptRoot 'check_versions.ps1')
if ($LASTEXITCODE) { throw "version check failed" }
$version = ($version | Select-Object -Last 1).Trim()
Info "Canonical version: $version"

# ---------- Reset dist ----------
if ($Clean -and (Test-Path $LightDist)) {
    Step "Cleaning $LightDist"
    Remove-Item -Recurse -Force $LightDist
}
New-Item -ItemType Directory -Force -Path $LightDist, $VendorDir, $AppDir, (Join-Path $LightDist 'runtime') | Out-Null

# ---------- C2. git archive source ----------
Step "Exporting tracked source via git archive"
$tar = Join-Path $BuildRoot 'light-source.tar'
if (Test-Path $tar) { Remove-Item -Force $tar }

Push-Location $RepoRoot
try {
    & git archive --format=tar -o $tar HEAD writansub
    if ($LASTEXITCODE) { throw "git archive failed" }
} finally {
    Pop-Location
}
Push-Location $AppDir
try {
    & tar --force-local -xf $tar
    if ($LASTEXITCODE) { throw "tar extract failed" }
} finally {
    Pop-Location
}
Remove-Item -Force $tar
Info ("Packaged {0} source files" -f (Get-ChildItem -Recurse (Join-Path $AppDir 'writansub') -File).Count)

# ---------- C3. Native wheel ----------
if (-not $SkipNative) {
    Step "Building writansub_native wheel"
    $buildPy = & uv python find 3.12 2>$null
    if (-not $buildPy -or $LASTEXITCODE) {
        throw "No Python 3.12 found for native wheel build. 'uv python install 3.12' or python.org."
    }
    $buildPy = ($buildPy | Select-Object -Last 1).Trim()

    Push-Location (Join-Path $RepoRoot 'native')
    try {
        & uv tool run --from 'maturin>=1.4.0' maturin build --release --out $VendorDir --interpreter $buildPy
        if ($LASTEXITCODE) { throw "maturin build failed" }
    } finally {
        Pop-Location
    }

    $nativeCargo = Get-Content (Join-Path $RepoRoot 'native\Cargo.toml') -Raw
    if ($nativeCargo -notmatch '(?ms)^\s*\[package\].*?^\s*version\s*=\s*"([^"]+)"') {
        throw "Cannot parse native Cargo.toml [package] version"
    }
    $nativeVer = $matches[1]
    $nativeWheel = Get-ChildItem $VendorDir -Filter "writansub_native-$nativeVer-*.whl" | Select-Object -First 1
    if (-not $nativeWheel) {
        throw "native wheel matching 'writansub_native-$nativeVer-*.whl' not produced"
    }
    Info "Native wheel: $($nativeWheel.Name)"
}

# ---------- C4. Export requirements lock ----------
Step "Exporting requirements.lock.txt"
$reqFile = Join-Path $LightDist 'requirements.lock.txt'
Push-Location $RepoRoot
try {
    & uv export --frozen --no-hashes `
        --no-emit-project `
        --no-emit-package writansub_native `
        --format requirements-txt `
        --output-file $reqFile
    if ($LASTEXITCODE) { throw "uv export failed" }
} finally {
    Pop-Location
}

# ---------- C5. Bundle embeddable zip ----------
Step "Bundling Python embeddable"
$embedZip = Join-Path $Cache "python-$PythonVer-embed-amd64.zip"
if (-not (Test-Path $embedZip)) {
    Info "Downloading $PythonVer embeddable"
    if (-not (Test-Path $Cache)) { New-Item -ItemType Directory -Force -Path $Cache | Out-Null }
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest `
            -Uri "https://www.python.org/ftp/python/$PythonVer/python-$PythonVer-embed-amd64.zip" `
            -OutFile $embedZip -UseBasicParsing
    } finally {
        $ProgressPreference = 'Continue'
    }
}
Copy-Item $embedZip (Join-Path $VendorDir 'python-embed.zip') -Force

# ---------- C6. Emit install.bat ----------
Step "Emitting install.bat"
$installBat = @'
@echo off
setlocal enabledelayedexpansion
title WritanSub 安装 (轻量版)
cd /d "%~dp0"

echo.
echo ============================================================
echo   WritanSub 轻量版安装程序
echo ============================================================
echo 本安装器会:
echo   [1/5] 解压内嵌 Python (~10 MB)
echo   [2/5] 配置 runtime
echo   [3/5] 检查并安装 uv (Python 包管理器)
echo   [4/5] 从清华/官方镜像下载依赖 (~4 GB, 10-30 分钟)
echo   [5/5] 安装 native 扩展
echo.
echo 确保网络畅通。断网或镜像被屏蔽会触发镜像回退。
echo.
pause

:: -------- [1/5] 解压 embeddable --------
echo.
echo [1/5] 解压内嵌 Python ...
if exist "runtime\python.exe" (
    echo     runtime\python.exe 已存在, 跳过解压.
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Expand-Archive -LiteralPath 'vendor\python-embed.zip' -DestinationPath 'runtime' -Force"
    if errorlevel 1 (
        echo Python 解压失败.
        pause
        exit /b 1
    )
)

:: -------- [2/5] 配置 ._pth + site-packages --------
echo.
echo [2/5] 配置 runtime ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$f = Get-ChildItem 'runtime' -Filter 'python*._pth' | Select-Object -First 1; if (-not $f) { Write-Error 'no _pth'; exit 1 }; Set-Content $f.FullName -Value \"python312.zip`n.`nLib\site-packages`n..\app`nimport site\" -Encoding ASCII; New-Item -ItemType Directory -Force -Path 'runtime\Lib\site-packages' | Out-Null"
if errorlevel 1 (
    echo _pth 配置失败.
    pause
    exit /b 1
)

:: -------- [3/5] 确保 uv 可用 --------
echo.
echo [3/5] 检查 uv ...
where uv >nul 2>nul
if errorlevel 1 (
    echo     未检测到 uv, 正在从 astral.sh 安装 ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo uv 安装失败, 请检查网络.
        pause
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
) else (
    echo     uv 已就绪.
)

:: -------- [4/5] 装依赖 (3 次重试, 清华→官方 镜像回退) --------
echo.
echo [4/5] 下载并安装依赖 (约 4 GB) ...
set "MIRROR_TSINGHUA=https://pypi.tuna.tsinghua.edu.cn/simple"
set "MIRROR_PYPI=https://pypi.org/simple"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

set ATTEMPT=0
:retry_install
set /a ATTEMPT+=1
if %ATTEMPT% GTR 3 (
    echo 依赖安装在 3 次尝试后仍然失败. 请排查网络或代理.
    pause
    exit /b 1
)
if %ATTEMPT% EQU 1 (
    set "INDEX=%MIRROR_TSINGHUA%"
) else (
    set "INDEX=%MIRROR_PYPI%"
)
echo     尝试 %ATTEMPT%/3 - 镜像: !INDEX!
call uv pip install ^
    --python "runtime\python.exe" ^
    --target "runtime\Lib\site-packages" ^
    --no-deps ^
    --index-url "!INDEX!" ^
    --extra-index-url "%TORCH_INDEX%" ^
    --index-strategy unsafe-best-match ^
    -r "requirements.lock.txt"
if errorlevel 1 (
    echo     镜像 !INDEX! 失败, 重试 ...
    goto retry_install
)

:: -------- [5/5] native 扩展 --------
echo.
echo [5/5] 安装 native 扩展 ...
for %%f in ("vendor\writansub_native-*.whl") do (
    call uv pip install ^
        --python "runtime\python.exe" ^
        --target "runtime\Lib\site-packages" ^
        --reinstall --no-deps "%%f"
    if errorlevel 1 (
        echo native 扩展安装失败.
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo   安装完成! 双击 WritanSub.exe 启动 GUI.
echo   日志位置: %%APPDATA%%\WritanSub\launcher.log
echo ============================================================
pause
'@
$installBatCrlf = $installBat -replace "(?<!`r)`n", "`r`n"
[System.IO.File]::WriteAllText(
    (Join-Path $LightDist 'install.bat'),
    $installBatCrlf,
    [System.Text.Encoding]::GetEncoding(936)
)

# ---------- C7. README ----------
$readme = @"
WritanSub 轻量版 ($version)
==============================

这是 WritanSub 的"源码 + 安装脚本"版本, 本身只有约 30 MB.
所有第三方依赖 (torch / PySide6 等, 约 4 GB) 会在首次安装时下载.

系统要求
--------
- Windows 10/11 64 位
- NVIDIA 显卡, 驱动版本 >= 570.x (CUDA 12.8)
- 首次安装需要网络
- 磁盘至少 8 GB 可用空间

安装步骤
--------
1. 双击 install.bat
2. 等待依赖下载 (10-30 分钟, 视网速)
3. 安装完成后双击 WritanSub.exe 启动 GUI

命令行用法
----------
打开 cmd 切换到安装目录, 运行:
    WritanSubCLI.exe --help
    WritanSubCLI.exe pipeline <media>

日志
----
启动日志: %APPDATA%\WritanSub\launcher.log
用户配置: %APPDATA%\WritanSub\

如果镜像被屏蔽
--------------
install.bat 默认走清华镜像, 失败后自动切到 pypi.org.
如需固定镜像, 编辑 install.bat 里的 MIRROR_TSINGHUA / MIRROR_PYPI.

卸载
----
直接删除安装目录 (默认 %LocalAppData%\WritanSub) 即可.
用户配置留存在 %APPDATA%\WritanSub\, 如需完全清除也一并删除.
"@
[System.IO.File]::WriteAllText(
    (Join-Path $LightDist 'README.txt'),
    ($readme -replace "(?<!`r)`n", "`r`n"),
    (New-Object System.Text.UTF8Encoding($true))
)

# ---------- Rust launcher exes ----------
if (-not $SkipLauncher) {
    Step "Building Rust launcher"
    Push-Location (Join-Path $BuildRoot 'launcher')
    try {
        & cargo build --release
        if ($LASTEXITCODE) { throw "cargo build failed" }
    } finally {
        Pop-Location
    }
}
$launcherRelease = Join-Path $BuildRoot 'launcher\target\release'
Copy-Item -Force (Join-Path $launcherRelease 'WritanSub.exe')    (Join-Path $LightDist 'WritanSub.exe')
Copy-Item -Force (Join-Path $launcherRelease 'WritanSubCLI.exe') (Join-Path $LightDist 'WritanSubCLI.exe')

# ---------- Summary ----------
$size = (Get-ChildItem -Recurse $LightDist | Measure-Object Length -Sum).Sum / 1MB
Step ("Light bundle: {0:N1} MB at {1}" -f $size, $LightDist)
