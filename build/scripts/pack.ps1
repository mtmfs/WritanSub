#Requires -Version 5.1
<#
Pack build/dist/WritanSub into a single self-extracting installer:
  build/release/WritanSub-Setup-<Version>.exe

Uses 7-Zip lzma2 ultra + 7zSD.sfx module. If the final SFX exceeds the 2 GB
Github Release single-asset limit, emits a WARNING and prints the top-10
largest subdirectories inside runtime/ so the maintainer can prune further.

Requires 7-Zip installed at `C:\Program Files\7-Zip\` (or on PATH).
Install via `winget install 7zip.7zip`.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Version
)

$ErrorActionPreference = 'Stop'

$RepoRoot  = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$BuildRoot = Join-Path $RepoRoot 'build'
$AppDist   = Join-Path $BuildRoot 'dist\WritanSub'
$Runtime   = Join-Path $AppDist 'runtime'
$Release   = Join-Path $BuildRoot 'release'

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" -ForegroundColor DarkGray }

if (-not (Test-Path $AppDist)) {
    throw "No build output at $AppDist. Run build.ps1 first."
}

# Locate 7-Zip (skip the NVIDIA-bundled stub). Search common install roots
# across drives, then fall back to PATH.
$sevenZip = $null
$roots = @(
    'C:\Program Files\7-Zip',
    'C:\Program Files (x86)\7-Zip',
    'D:\7-Zip', 'E:\7-Zip', 'F:\7-Zip', 'G:\7-Zip'
)
foreach ($r in $roots) {
    $c = Join-Path $r '7z.exe'
    if (Test-Path $c) { $sevenZip = $c; break }
}
if (-not $sevenZip) {
    $onPath = Get-Command 7z -ErrorAction SilentlyContinue
    if ($onPath -and ($onPath.Source -notmatch 'NVIDIA')) {
        $sevenZip = $onPath.Source
    }
}
if (-not $sevenZip) {
    throw @"
7-Zip not found. Install via: winget install 7zip.7zip
If 7-Zip is installed on a non-system drive, add its path to PATH or edit this
script's `\$roots` list.
"@
}
Info "7-Zip: $sevenZip"

# Locate 7zSD.sfx. Modern 7-Zip (23+) no longer ships this GUI SFX module;
# fall back to the one we cache from 7-Zip 9.20 extras (lzma2-compatible).
$cacheDir = Join-Path $BuildRoot 'cache'
if (-not (Test-Path $cacheDir)) { New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null }

$sfxModule = $null
$sfxCandidates = @(
    (Join-Path (Split-Path $sevenZip -Parent) '7zSD.sfx'),
    (Join-Path $cacheDir '7zSD.sfx')
)
foreach ($c in $sfxCandidates) { if (Test-Path $c) { $sfxModule = $c; break } }

if (-not $sfxModule) {
    Info "Bootstrapping 7zSD.sfx from 7-Zip 9.20 extras"
    $extraArchive = Join-Path $cacheDir '7z920_extra.7z'
    if (-not (Test-Path $extraArchive)) {
        $ProgressPreference = 'SilentlyContinue'
        try {
            Invoke-WebRequest -Uri 'https://www.7-zip.org/a/7z920_extra.7z' -OutFile $extraArchive -UseBasicParsing
        } finally {
            $ProgressPreference = 'Continue'
        }
    }
    Push-Location $cacheDir
    try {
        & $sevenZip e $extraArchive '7zSD.sfx' -y | Out-Null
    } finally {
        Pop-Location
    }
    $sfxModule = Join-Path $cacheDir '7zSD.sfx'
    if (-not (Test-Path $sfxModule)) {
        throw "Failed to bootstrap 7zSD.sfx from extras archive"
    }
}
Info "SFX module: $sfxModule"

New-Item -ItemType Directory -Force -Path $Release | Out-Null

$sfxPath = Join-Path $Release "WritanSub-Setup-$Version.exe"
$tmp7z   = Join-Path $Release 'WritanSub.7z'
$cfgPath = Join-Path $Release 'sfx-config.txt'

foreach ($p in @($sfxPath, $tmp7z, $cfgPath)) {
    if (Test-Path $p) { Remove-Item -Force $p }
}

Step "Compressing $AppDist -> $tmp7z"
& $sevenZip a -t7z `
    '-mx=9' '-m0=lzma2:d=1024m:fb=273:mf=bt4:lc=4' `
    '-ms=on' '-mmt=on' `
    $tmp7z (Join-Path $AppDist '*') | Out-Host
if ($LASTEXITCODE) { throw "7z archive failed" }

$archiveSize = (Get-Item $tmp7z).Length / 1MB
Info ("Raw 7z archive: {0:N1} MB" -f $archiveSize)

# Build SFX config (UTF-8)
$sfxConfig = @"
;!@Install@!UTF-8!
Title="WritanSub $Version"
BeginPrompt="解压 WritanSub 到目标目录?"
ExtractTitle="WritanSub 安装"
ExtractPathText="安装目录"
ExtractDialogText="正在解压, 约需 1-3 分钟..."
InstallPath="%LocalAppData%\\WritanSub"
ExecuteFile="WritanSub.exe"
GUIFlags="64+8"
OverwriteMode="0"
;!@InstallEnd@!
"@
Set-Content -Path $cfgPath -Value $sfxConfig -Encoding UTF8

Step "Assembling SFX installer"
# Binary-concatenate: sfx-module + config + archive -> output exe
cmd /c copy /b "`"$sfxModule`"" + "`"$cfgPath`"" + "`"$tmp7z`"" "`"$sfxPath`"" | Out-Null
if (-not (Test-Path $sfxPath)) { throw "SFX build failed — output missing" }

Remove-Item -Force $tmp7z, $cfgPath

$sfxSizeBytes = (Get-Item $sfxPath).Length
$sfxSizeMB = $sfxSizeBytes / 1MB
Step ("Built {0} ({1:N1} MB)" -f $sfxPath, $sfxSizeMB)

if ($sfxSizeBytes -gt 2GB) {
    Write-Warning ("SFX exceeds Github Release 2GB single-asset limit: {0:N1} MB" -f $sfxSizeMB)
    Write-Host ""
    Write-Host "Top 10 largest subdirectories in runtime/:" -ForegroundColor Yellow
    Get-ChildItem $Runtime -Directory -Force -Recurse |
        Where-Object { $_.FullName -notmatch '__pycache__' } |
        ForEach-Object {
            [PSCustomObject]@{
                Path = $_.FullName.Substring($Runtime.Length + 1)
                Size = (Get-ChildItem -Recurse -File -Force $_.FullName -ErrorAction SilentlyContinue |
                    Measure-Object Length -Sum).Sum
            }
        } |
        Where-Object { $_.Size -gt 0 } |
        Sort-Object Size -Descending |
        Select-Object -First 10 |
        ForEach-Object { Write-Host ("    {0,-60} {1,10:N1} MB" -f $_.Path, ($_.Size / 1MB)) -ForegroundColor DarkYellow }
    Write-Host ""
    Write-Warning "Fallback options:"
    Write-Warning "  1. Tighten prune list in build.ps1 (Qt translations, scipy tests, ...)"
    Write-Warning "  2. Ship $sfxPath via cloud drive (OneDrive / Baidu) and link from release notes"
}
