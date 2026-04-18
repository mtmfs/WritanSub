#Requires -Version 5.1
<#
Release orchestrator.

Examples:
  .\build\scripts\release.ps1 -Full -Clean
  .\build\scripts\release.ps1 -Light -Clean
  .\build\scripts\release.ps1 -Both -Clean

Flow:
  1. Version alignment check (aborts on drift)
  2. Warn on dirty release sources (git archive exports HEAD, not working tree)
  3. Full flavour:  build.ps1 (with smoke test) → pack.ps1 (7z SFX, 2GB warning)
  4. Light flavour: light.ps1 → ISCC (Inno Setup) compile
  5. Print artifact sizes + manual verification checklist
#>

[CmdletBinding()]
param(
    [switch]$Light,
    [switch]$Full,
    [switch]$Both,
    [switch]$Clean,
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$BuildRoot  = Join-Path $RepoRoot 'build'
$IscPath    = Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'
$IssFile    = Join-Path $BuildRoot 'installer\writansub.iss'
$Release    = Join-Path $BuildRoot 'release'

if ($Both) { $Light = $true; $Full = $true }
if (-not ($Light -or $Full)) {
    throw "Specify one of: -Light, -Full, -Both"
}

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" -ForegroundColor DarkGray }
function Warn($m) { Write-Warning $m }

# ---------- 1. Version check ----------
Step "Verifying version alignment"
$version = & (Join-Path $PSScriptRoot 'check_versions.ps1')
if ($LASTEXITCODE) { throw "version check failed" }
$version = ($version | Select-Object -Last 1).Trim()
Info "Release version: $version"

# ---------- 2. Dirty tree warning ----------
Push-Location $RepoRoot
try {
    $dirty = & git status --porcelain pyproject.toml writansub native build/launcher 2>$null
    if ($dirty) {
        Warn "Uncommitted changes in release sources (git archive exports HEAD only):"
        Write-Host $dirty -ForegroundColor Yellow
        Write-Host ""
    }
} finally {
    Pop-Location
}

New-Item -ItemType Directory -Force -Path $Release | Out-Null

# ---------- 3. Full ----------
if ($Full) {
    Step "Building full offline bundle"
    $buildArgs = @{}
    if ($Clean)     { $buildArgs['Clean']     = $true }
    if ($SkipSmoke) { $buildArgs['SkipSmoke'] = $true }
    & (Join-Path $PSScriptRoot 'build.ps1') @buildArgs
    if ($LASTEXITCODE) { throw "build.ps1 failed" }

    if (-not (Test-Path $IscPath)) {
        throw "Inno Setup not found at $IscPath. Install via: winget install JRSoftware.InnoSetup"
    }
    Step "Compiling Inno Setup (full, DiskSpanning)"
    & $IscPath "/DAppVersion=$version" $IssFile
    if ($LASTEXITCODE) { throw "Inno Setup compile (full) failed" }
}

# ---------- 4. Light ----------
if ($Light) {
    Step "Building lite source bundle"
    $lightArgs = @{}
    if ($Clean) { $lightArgs['Clean'] = $true }
    & (Join-Path $PSScriptRoot 'light.ps1') @lightArgs
    if ($LASTEXITCODE) { throw "light.ps1 failed" }

    if (-not (Test-Path $IscPath)) {
        throw "Inno Setup not found at $IscPath. Install via: winget install JRSoftware.InnoSetup"
    }
    Step "Compiling Inno Setup (lite)"
    & $IscPath "/DAppVersion=$version" $IssFile
    if ($LASTEXITCODE) { throw "Inno Setup compile failed" }
}

# ---------- 5. Summary ----------
Write-Host ""
Step "Release artifacts"
$matched = Get-ChildItem $Release -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match [regex]::Escape($version) }
if (-not $matched) {
    Warn "No artifacts found in $Release matching version $version"
} else {
    foreach ($f in $matched) {
        $mb = $f.Length / 1MB
        $tag = ''
        if ($f.Length -gt 2GB) { $tag = '  [EXCEEDS 2GB — needs cloud-drive link]' }
        Write-Host ("    {0,-55} {1,10:N1} MB{2}" -f $f.Name, $mb, $tag) -ForegroundColor White
    }
}

Write-Host ""
Step "Manual verification required (none of this is automated)"
Write-Host "  1. Copy artifact to a clean Windows 11 VM without Python/uv installed" -ForegroundColor Yellow
Write-Host "  2. Full bundle: double-click the SFX, extract, run WritanSub.exe" -ForegroundColor Yellow
Write-Host "  3. Lite bundle: run Inno installer, let install.bat run, then launch WritanSub.exe" -ForegroundColor Yellow
Write-Host "  4. Run one short transcription end-to-end" -ForegroundColor Yellow
Write-Host "  5. Check %APPDATA%\WritanSub\launcher.log for warnings/tracebacks" -ForegroundColor Yellow
