# Install arxiv-digest and add it to the Start menu. Run it again to update.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SampsonML/arxiv-digest/main/install.ps1 | iex"
#   powershell -ExecutionPolicy ByPass -File install.ps1    from a clone: install that clone's code
#
# It installs uv (https://docs.astral.sh/uv/) if needed, then arxiv-digest as a
# uv tool (its own Python environment), using the exact package versions in uv.lock.
#
# Options: --no-open (or $env:ARXIV_DIGEST_NO_OPEN = "1") skips opening the app.
# $env:ARXIV_DIGEST_REF picks a branch or tag (default: main), and
# $env:ARXIV_DIGEST_ARCHIVE a .zip of the source to install instead (tests).

function Install-ArxivDigest {
    param([string]$Here, [switch]$NoOpen)

    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is very slow with a progress bar
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $repo = 'SampsonML/arxiv-digest'
    $command = "powershell -ExecutionPolicy ByPass -c `"irm https://raw.githubusercontent.com/$repo/main/install.ps1 | iex`""

    function Find-Uv {
        $found = Get-Command uv -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
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
            Write-Host 'uv is needed. Install it yourself, or see "Other ways to install" in the README.'
            return
        }
        powershell -NoProfile -ExecutionPolicy ByPass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
        $uv = Find-Uv
        if (-not $uv) {
            Write-Host "uv was installed, but can't be found. Open a new PowerShell window and run this again."
            return
        }
    }

    # From a clone, install the code next to this script.
    $source = $null
    if ($Here) {
        $pyproject = Join-Path $Here 'pyproject.toml'
        if ((Test-Path -LiteralPath $pyproject) -and (Select-String -LiteralPath $pyproject -Pattern '^name = "arxiv-digest"' -Quiet)) {
            $source = $Here
        }
    }

    $work = Join-Path ([IO.Path]::GetTempPath()) ('arxiv-digest-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work | Out-Null
    try {
        $fromClone = [bool]$source
        if (-not $source) {
            $ref = if ($env:ARXIV_DIGEST_REF) { $env:ARXIV_DIGEST_REF } else { 'main' }
            $archive = if ($env:ARXIV_DIGEST_ARCHIVE) { $env:ARXIV_DIGEST_ARCHIVE } else { "https://github.com/$repo/archive/$ref.zip" }
            Write-Host "Downloading arxiv-digest ($ref) from GitHub..."
            $zip = Join-Path $work 'source.zip'
            if ($archive -match '^https?://') {
                Invoke-WebRequest -UseBasicParsing -Uri $archive -OutFile $zip
            } else {
                Copy-Item -LiteralPath $archive -Destination $zip
            }
            $unpacked = Join-Path $work 'source'
            Expand-Archive -LiteralPath $zip -DestinationPath $unpacked
            $source = (Get-ChildItem -LiteralPath $unpacked -Directory | Select-Object -First 1).FullName
        }

        Write-Host 'Installing arxiv-digest (the first time takes a minute)...'
        # The exact versions in uv.lock, so everyone gets the tested set of packages.
        $constraints = Join-Path $work 'constraints.txt'
        & $uv export --quiet --frozen --no-dev --no-hashes --no-emit-project --directory $source --output-file $constraints
        if ($LASTEXITCODE -ne 0) { throw 'uv export failed' }
        & $uv tool install --python '>=3.11,<3.15' --reinstall-package arxiv-digest --constraints $constraints $source
        if ($LASTEXITCODE -ne 0) { throw 'uv tool install failed' }

        $bin = (& $uv tool dir --bin | Out-String).Trim()
        $setupArgs = @('setup')
        if ($NoOpen) { $setupArgs += '--no-open' }
        & (Join-Path $bin 'arxiv-digest.exe') @setupArgs
        if ($LASTEXITCODE -ne 0) { throw 'arxiv-digest setup failed' }

        Write-Host ''
        if ($fromClone) {
            Write-Host 'To update later: git pull, then run install.ps1 again.'
        } else {
            Write-Host 'To update arxiv-digest later, run the same command again:'
            Write-Host "  $command"
        }
        $onPath = ($env:PATH -split ';') | Where-Object { $_.TrimEnd('\') -ieq $bin.TrimEnd('\') }
        if (-not $onPath) {
            Write-Host 'To use the arxiv-digest command in a terminal, run "uv tool update-shell" and open a new window.'
        }
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# No `exit` anywhere: when piped into iex, exit would close the user's window.
$here = if ($PSScriptRoot) { $PSScriptRoot } else { $null }
$noOpen = ($args -contains '--no-open') -or ($env:ARXIV_DIGEST_NO_OPEN -eq '1')
Install-ArxivDigest -Here $here -NoOpen:$noOpen
