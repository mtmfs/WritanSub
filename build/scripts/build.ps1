#Requires -Version 5.1
<#
Build the full offline bundle at build/dist/WritanSub/.

Layout produced:
  WritanSub/
    WritanSub.exe          <- Rust GUI launcher (runtime\pythonw.exe)
    WritanSubCLI.exe       <- Rust CLI launcher  (runtime\python.exe)
    app/writansub/         <- git archive HEAD
    runtime/               <- embeddable Python + deps in Lib\site-packages

Unlike the old `uv venv --relocatable` approach, runtime/ is a true self-contained
Python installation (official Windows embeddable distribution). No pyvenv.cfg,
no redirector exes, no base Python required on target machines.
#>

[CmdletBinding()]
param(
    [switch]$SkipLauncher,
    [switch]$Clean,
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'

$RepoRoot  = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$BuildRoot = Join-Path $RepoRoot 'build'
$DistRoot  = Join-Path $BuildRoot 'dist'
$AppDist   = Join-Path $DistRoot 'WritanSub'
$Runtime   = Join-Path $AppDist 'runtime'
$AppDir    = Join-Path $AppDist 'app'
$ReqFile   = Join-Path $BuildRoot 'requirements.lock.txt'

# rustup toolchain bin for cargo (launcher build)
$RustupBin = Join-Path $env:USERPROFILE '.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin'
if ((Test-Path $RustupBin) -and ($env:PATH -notlike "*$RustupBin*")) {
    $env:PATH = "$RustupBin;$env:PATH"
}

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" -ForegroundColor DarkGray }

# ---------- B1. Version alignment ----------
Step "Checking version alignment"
$version = & (Join-Path $PSScriptRoot 'check_versions.ps1')
if ($LASTEXITCODE) { throw "version check failed" }
$version = ($version | Select-Object -Last 1).Trim()
Info "Canonical version: $version"

# ---------- B2. Materialise runtime ----------
if ($Clean -and (Test-Path $AppDist)) {
    Step "Cleaning $AppDist"
    Remove-Item -Recurse -Force $AppDist
}
New-Item -ItemType Directory -Force -Path $AppDist | Out-Null

Step "Creating embeddable runtime at $Runtime"
& (Join-Path $PSScriptRoot 'make_runtime.ps1') -TargetPath $Runtime | Out-Null

$runtimePython = Join-Path $Runtime 'python.exe'
$sitePkg       = Join-Path $Runtime 'Lib\site-packages'

# ---------- B3. Export & install locked deps ----------
Step "Exporting locked dependencies to $ReqFile"
Push-Location $RepoRoot
try {
    & uv export --frozen --no-hashes `
        --no-emit-project `
        --format requirements-txt `
        --output-file $ReqFile
    if ($LASTEXITCODE) { throw "uv export failed" }
} finally {
    Pop-Location
}

Step "Installing dependencies into embeddable runtime"
& uv pip install `
    --python $runtimePython `
    --no-deps `
    --index-url https://pypi.org/simple `
    --extra-index-url https://download.pytorch.org/whl/cu128 `
    --index-strategy unsafe-best-match `
    -r $ReqFile
if ($LASTEXITCODE) { throw "uv pip install into embeddable failed" }

# ---------- B4. App source via git archive ----------
Step "Exporting tracked source via git archive"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
$tar = Join-Path $BuildRoot 'app-source.tar'
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
    # GNU tar 处理含盘符路径需 --force-local；System32 bsdtar 不认该参数但原生支持盘符
    $tarIsGnu = ((& tar --version 2>$null | Select-Object -First 1) -match 'GNU tar')
    if ($tarIsGnu) { & tar --force-local -xf $tar } else { & tar -xf $tar }
    if ($LASTEXITCODE) { throw "tar extract failed" }
} finally {
    Pop-Location
}
Remove-Item -Force $tar
$srcCount = (Get-ChildItem -Recurse (Join-Path $AppDir 'writansub') -File).Count
Info "Packaged $srcCount source files from HEAD"

# ---------- B5. Rust launcher exes ----------
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
Copy-Item -Force (Join-Path $launcherRelease 'WritanSub.exe')    (Join-Path $AppDist 'WritanSub.exe')
Copy-Item -Force (Join-Path $launcherRelease 'WritanSubCLI.exe') (Join-Path $AppDist 'WritanSubCLI.exe')
Info "Launchers copied to $AppDist"

# ---------- B6. Prune runtime ----------
Step "Pruning runtime to reduce size"
# __pycache__
Get-ChildItem -Recurse $sitePkg -Directory -Force -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
# bytecode files
Get-ChildItem -Recurse $sitePkg -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq '.pyc' -or $_.Extension -eq '.pyo' } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# tests/docs/examples (keep torch/numpy untouched to avoid breakage)
$junkDirs = @('tests','test','docs','examples','benchmarks','__pypackages__')
foreach ($d in $junkDirs) {
    Get-ChildItem -Recurse $sitePkg -Directory -Force -Filter $d -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\(torch|numpy|sympy)\\' } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# torch: drop test data and C++ headers not needed at runtime
$torchDir = Join-Path $sitePkg 'torch'
foreach ($sub in @('test','include')) {
    $p = Join-Path $torchDir $sub
    if (Test-Path $p) { Remove-Item -Recurse -Force $p -ErrorAction SilentlyContinue }
}

# ---------- B7. Smoke test ----------
if (-not $SkipSmoke) {
    Step "Running smoke test against built runtime"

    $smokeScript = @"
import os, sys
sys.path.insert(0, os.environ['WRITANSUB_APP_DIR'])
import writansub
from writansub.gui.app import MainWindow
print('smoke-ok', writansub.__version__)
"@
    $env:WRITANSUB_APP_DIR = $AppDir
    $env:QT_QPA_PLATFORM = 'minimal'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    try {
        & $runtimePython -c $smokeScript
        $smokeCode = $LASTEXITCODE
    } finally {
        Remove-Item Env:\WRITANSUB_APP_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:\QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        Remove-Item Env:\PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
    }
    if ($smokeCode) {
        throw "SMOKE FAILED: runtime cannot import writansub + MainWindow (exit=$smokeCode)"
    }
    Info "Smoke test passed"
}

# ---------- Summary ----------
$size = (Get-ChildItem -Recurse $AppDist | Measure-Object Length -Sum).Sum / 1MB
Step ("Build complete: {0:N0} MB at {1}" -f $size, $AppDist)
Info "Next: run pack.ps1 -Version $version to produce the SFX installer"
