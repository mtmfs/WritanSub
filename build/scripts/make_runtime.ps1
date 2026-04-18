#Requires -Version 5.1
<#
Materialise an embeddable-Python runtime/ at the given path.

Produces a self-contained directory containing:
  python.exe / pythonw.exe       <- official Windows embeddable (real interpreters, not uv redirectors)
  python312.dll / python3.dll    <- interpreter core
  python312.zip                  <- stdlib
  python312._pth                 <- patched to enable site.py + append Lib\site-packages
  Lib\site-packages\             <- empty, ready for `uv pip install --python ...`

Idempotent: if TargetPath exists it is nuked and recreated. The embeddable zip is
cached under CacheDir so repeated runs don't re-download.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$TargetPath,

    [string]$PythonVersion = '3.12.10',

    [string]$CacheDir
)

$ErrorActionPreference = 'Stop'

if (-not $CacheDir) {
    $CacheDir = Join-Path $PSScriptRoot '..\cache'
}

function Step($m) { Write-Host "    [runtime] $m" -ForegroundColor DarkGray }

$zipName  = "python-$PythonVersion-embed-amd64.zip"
$zipUrl   = "https://www.python.org/ftp/python/$PythonVersion/$zipName"
$cachedZip = Join-Path $CacheDir $zipName

# 1. Download if missing
if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
}
if (-not (Test-Path $cachedZip)) {
    Step "Downloading $zipName"
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $zipUrl -OutFile $cachedZip -UseBasicParsing
    } finally {
        $ProgressPreference = 'Continue'
    }
    if (-not (Test-Path $cachedZip)) {
        throw "Download failed: $zipUrl"
    }
} else {
    Step "Using cached $zipName"
}

# 2. Extract (nuke target first for idempotency)
if (Test-Path $TargetPath) {
    Step "Wiping existing $TargetPath"
    Remove-Item -Recurse -Force $TargetPath
}
New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null
Step "Extracting to $TargetPath"
Expand-Archive -LiteralPath $cachedZip -DestinationPath $TargetPath -Force

# 3. Patch ._pth
$pthFile = Get-ChildItem $TargetPath -Filter 'python*._pth' | Select-Object -First 1
if (-not $pthFile) {
    throw "No python*._pth found in embeddable archive — layout changed?"
}
Step "Patching $($pthFile.Name)"
$newPth = @"
python312.zip
.
Lib\site-packages
import site
"@
Set-Content -Path $pthFile.FullName -Value $newPth -Encoding ASCII

# 4. Create empty Lib\site-packages
$sitePkg = Join-Path $TargetPath 'Lib\site-packages'
New-Item -ItemType Directory -Force -Path $sitePkg | Out-Null

# 5. Sanity check: python.exe must exist
$py = Join-Path $TargetPath 'python.exe'
$pyw = Join-Path $TargetPath 'pythonw.exe'
if (-not (Test-Path $py))  { throw "python.exe missing after extraction"  }
if (-not (Test-Path $pyw)) { throw "pythonw.exe missing after extraction" }

Step "Runtime ready: $TargetPath"
return (Resolve-Path $TargetPath).Path
