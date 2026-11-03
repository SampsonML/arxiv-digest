# Install arxiv-digest from this folder and add it to the Start menu.
# Run it again after `git pull` to update. In PowerShell, from this folder:
#
#   powershell -ExecutionPolicy Bypass -File install.ps1             install, add the app, and open it
#   powershell -ExecutionPolicy Bypass -File install.ps1 --no-open   the same, without opening the browser
#
# It uses uv (https://docs.astral.sh/uv/) to set up a supported Python and the
# packages in .venv inside this folder; if uv is missing, it offers to install it.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    # Where uv's installer puts it, for when this window's PATH doesn't have it yet.
    foreach ($candidate in @(
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
            (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe'))) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Host 'arxiv-digest uses uv (https://docs.astral.sh/uv/) to set up Python and its packages.'
    $answer = Read-Host 'Install uv now? [Y/n]'
    if ($answer -match '^[nN]') {
        Write-Host 'Install uv yourself, or see "Other ways to install" in README.md.'
        exit 1
    }
    powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    $uv = Find-Uv
    if (-not $uv) {
        Write-Host "uv was installed, but this script can't find it. Open a new PowerShell window and run the installer again."
        exit 1
    }
}

Write-Host "Setting up arxiv-digest in $PSScriptRoot\.venv (the first time takes a minute)..."
& $uv sync --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m arxiv_digest setup @args
exit $LASTEXITCODE
