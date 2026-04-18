#Requires -Version 5.1
<#
Enforce version alignment across release sources.

Rules:
  - pyproject.toml (project.version)   == writansub/__init__.py (__version__)
  - native/Cargo.toml                  == native/pyproject.toml (must match each other exactly)
  - major.minor of native              == major.minor of pyproject
  - major.minor of build/launcher/Cargo.toml == major.minor of pyproject

Writes the canonical version (pyproject) to stdout so callers can capture it with:
  $ver = & .\check_versions.ps1

Any mismatch throws, aborting the pipeline.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Parse-PyprojectVersion([string]$path) {
    $t = Get-Content $path -Raw
    if ($t -match '(?ms)^\s*\[project\].*?^\s*version\s*=\s*"([^"]+)"') {
        return $matches[1]
    }
    if ($t -match '(?ms)^\s*\[package\].*?^\s*version\s*=\s*"([^"]+)"') {
        return $matches[1]
    }
    throw "No version found in $path"
}

function Parse-InitVersion([string]$path) {
    $t = Get-Content $path -Raw
    if ($t -match '__version__\s*=\s*"([^"]+)"') {
        return $matches[1]
    }
    throw "No __version__ found in $path"
}

function Major-Minor([string]$v) {
    $parts = $v -split '\.'
    if ($parts.Count -lt 2) { throw "Version '$v' has no major.minor" }
    return "$($parts[0]).$($parts[1])"
}

$vPyproject = Parse-PyprojectVersion (Join-Path $RepoRoot 'pyproject.toml')
$vInit      = Parse-InitVersion     (Join-Path $RepoRoot 'writansub\__init__.py')
$vNativeCg  = Parse-PyprojectVersion (Join-Path $RepoRoot 'native\Cargo.toml')
$vNativePy  = Parse-PyprojectVersion (Join-Path $RepoRoot 'native\pyproject.toml')
$vLauncher  = Parse-PyprojectVersion (Join-Path $RepoRoot 'build\launcher\Cargo.toml')

if ($vPyproject -ne $vInit) {
    throw "version mismatch: pyproject.toml ($vPyproject) != writansub/__init__.py ($vInit)"
}
if ($vNativeCg -ne $vNativePy) {
    throw "version mismatch: native/Cargo.toml ($vNativeCg) != native/pyproject.toml ($vNativePy)"
}

$mmPy = Major-Minor $vPyproject
if ((Major-Minor $vNativeCg) -ne $mmPy) {
    throw "native major.minor ($vNativeCg) must match pyproject ($vPyproject)"
}
if ((Major-Minor $vLauncher) -ne $mmPy) {
    throw "launcher major.minor ($vLauncher) must match pyproject ($vPyproject)"
}

Write-Host "    [versions] pyproject=$vPyproject init=$vInit native=$vNativeCg launcher=$vLauncher" -ForegroundColor DarkGray
Write-Output $vPyproject
