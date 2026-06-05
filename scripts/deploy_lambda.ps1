# Deploy de las Lambdas de cloud (src/cloud/*.py).
#
# Usage:
#   .\scripts\deploy_lambda.ps1 [-Environment dev] [-Function persist_event|query_aggregates|ingest_pos_transaction]
#
# Default -Function persist_event (back-compat). Cada CFN update resetea las 3
# Lambdas al ZipFile placeholder → hay que redeployar las tres con este script.
#
# Equivalente PowerShell de scripts/deploy_lambda.sh. Empaqueta el .py +
# psycopg binary (target manylinux2014_x86_64, no Windows) en un zip y
# actualiza la function code.
#
# Nota: NO usamos $ErrorActionPreference="Stop" porque PS 5.1 trata stderr
# de procesos nativos (pip notices, aws cli warnings) como NativeCommandError
# y aborta. En su lugar, validamos $LASTEXITCODE despues de cada comando
# nativo y -ErrorAction Stop per-comando en cmdlets de PowerShell.

[CmdletBinding()]
param(
    [string]$Environment = "dev",
    [ValidateSet("persist_event", "query_aggregates", "ingest_pos_transaction")]
    [string]$Function = "persist_event"
)

function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

# Mapa función → (nombre del Lambda en AWS, source file). El handler de cada
# Lambda es <modulo>.handler, así que el .py se empaqueta con su nombre original.
$lambdaMap = @{
    persist_event          = @{ Fn = "people-counter-persist-event-$Environment";   Src = "src\cloud\persist_event.py" }
    query_aggregates       = @{ Fn = "people-counter-query-aggregates-$Environment"; Src = "src\cloud\query_aggregates.py" }
    ingest_pos_transaction = @{ Fn = "people-counter-ingest-pos-$Environment";       Src = "src\cloud\ingest_pos_transaction.py" }
}
$functionName = $lambdaMap[$Function].Fn
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$srcFile = Join-Path $repoRoot $lambdaMap[$Function].Src

if (-not (Test-Path $srcFile)) {
    Fail "$srcFile not found"
}

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "lambda-deploy-$([System.Guid]::NewGuid().ToString('N').Substring(0,8))"
New-Item -ItemType Directory -Path $tmpDir -ErrorAction Stop | Out-Null

try {
    Write-Host "Copying source ($Function) -> $tmpDir"
    Copy-Item $srcFile (Join-Path $tmpDir (Split-Path $srcFile -Leaf)) -ErrorAction Stop

    # Detectar el binary de Python (py, python3, python — primero el que responda).
    $pythonExe = $null
    foreach ($candidate in @("py", "python3", "python")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $pythonExe = $cmd.Source; break }
    }
    if (-not $pythonExe) {
        Fail "No encontre Python (probe py, python3, python). Instala Python 3.10+ desde python.org."
    }
    Write-Host "Using Python: $pythonExe"

    Write-Host "Installing psycopg[binary] for Lambda runtime (manylinux2014_x86_64)..."
    & $pythonExe -m pip install `
        --platform manylinux2014_x86_64 `
        --target $tmpDir `
        --implementation cp `
        --python-version 3.13 `
        --only-binary=:all: `
        --upgrade `
        "psycopg[binary]==3.2.*"
    if ($LASTEXITCODE -ne 0) {
        Fail "pip install failed (exit $LASTEXITCODE)"
    }

    $zipPath = Join-Path ([System.IO.Path]::GetTempPath()) "lambda-$([System.Guid]::NewGuid().ToString('N').Substring(0,8)).zip"
    Write-Host "Creating zip at $zipPath ..."
    # Comprimir TODO el contenido del tmpDir (no el dir mismo).
    Push-Location $tmpDir
    try {
        Compress-Archive -Path .\* -DestinationPath $zipPath -Force -ErrorAction Stop
    } finally {
        Pop-Location
    }

    $sizeKB = [math]::Round((Get-Item $zipPath).Length / 1024)
    Write-Host "Package size: $sizeKB KB"

    Write-Host "Deploying $functionName..."
    # NO usamos --publish para evitar que se acumulen Lambda versions
    # numeradas a cada deploy. Para PoC con 1 device la IoT Rule invoca
    # $LATEST directamente, no hace falta versioning.
    aws lambda update-function-code `
        --function-name $functionName `
        --zip-file "fileb://$zipPath" `
        --output table `
        --query '{Function:FunctionName,LastModified:LastModified,Size:CodeSize}'
    if ($LASTEXITCODE -ne 0) {
        Fail "aws lambda update-function-code failed (exit $LASTEXITCODE)"
    }

    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
} finally {
    Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
