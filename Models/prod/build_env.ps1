$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ModelsRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile = Join-Path $PSScriptRoot "environment.yml"
$EnvName = "derma-env"
$KernelDisplayName = "Python 3.12 ($EnvName)"
$LogFile = Join-Path $PSScriptRoot "build_env.log"

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO", "WARN", "ERROR")]
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8

    switch ($Level) {
        "WARN"  { Write-Host $line -ForegroundColor Yellow }
        "ERROR" { Write-Host $line -ForegroundColor Red }
        default { Write-Host $line }
    }
}

"" | Out-File -FilePath $LogFile -Encoding UTF8
Write-Log "Starting environment setup. ProjectRoot='$ProjectRoot'"

if (-not (Test-Path $EnvFile)) {
    Write-Log "Environment file not found: $EnvFile" -Level "ERROR"
    throw "Environment file not found: $EnvFile"
}

$EnvExists = $false
$EnvListJson = conda env list --json 2>$null
if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($EnvListJson)) {
    $ParsedEnvList = $EnvListJson | ConvertFrom-Json
    foreach ($EnvPath in $ParsedEnvList.envs) {
        if ((Split-Path -Leaf $EnvPath) -eq $EnvName) {
            $EnvExists = $true
            break
        }
    }
}

if ($EnvExists) {
    Write-Log "Conda environment '$EnvName' already exists. Updating from '$EnvFile'..."
    conda env update -n $EnvName -f $EnvFile --prune
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Failed to update Conda environment '$EnvName'." -Level "ERROR"
        throw "conda env update failed."
    }
}
else {
    Write-Log "Creating Conda environment '$EnvName' from '$EnvFile'..."
    conda env create -f $EnvFile
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Failed to create Conda environment '$EnvName'." -Level "ERROR"
        throw "conda env create failed."
    }
}

Write-Log "Detecting Python interpreter path for '$EnvName'..."
$PythonExe = (
    conda run -n $EnvName python -c "import sys; print(sys.executable)" |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -Last 1
).Trim()

if (-not (Test-Path $PythonExe)) {
    Write-Log "Python interpreter not found: $PythonExe" -Level "ERROR"
    throw "Python interpreter not found: $PythonExe"
}

$PythonPathFile = Join-Path $PSScriptRoot ".derma-python-exe"
Set-Content -Path $PythonPathFile -Value $PythonExe -Encoding UTF8
Write-Log "Saved Python interpreter path to '$PythonPathFile': $PythonExe"

Write-Log "Registering Jupyter kernel '$KernelDisplayName'..."
conda run -n $EnvName python -m ipykernel install `
    --user `
    --name $EnvName `
    --display-name $KernelDisplayName
if ($LASTEXITCODE -ne 0) {
    Write-Log "Failed to register Jupyter kernel." -Level "ERROR"
    throw "ipykernel install failed."
}

$DotEnvFile = Join-Path $ModelsRoot ".env"
if (-not (Test-Path $DotEnvFile)) {
@"
DERMA_API_URL=
DERMA_EMAIL=
DERMA_PASSWORD=
DERMA_API_KEY=
SAMPLE_IMAGE_PATH=
"@ | Out-File -FilePath $DotEnvFile -Encoding utf8

    Write-Log "Created .env template at '$DotEnvFile'."
}
else {
    Write-Log ".env file already exists at '$DotEnvFile'. Skipping." -Level "WARN"
}

Write-Log "Environment setup completed! Environment='$EnvName', Python='$PythonExe'"