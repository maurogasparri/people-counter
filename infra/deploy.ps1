<#
.SYNOPSIS
    Deploy completo del stack people-counter en AWS.

.DESCRIPTION
    Orquesta el deploy en 2 fases del CFN + push de imagen Grafana a ECR +
    asociación del custom domain de App Runner. Pausa cuando necesita acción
    manual del usuario (correr SQL desde DBeaver + agregar CNAMEs al DNS
    provider externo).

.PARAMETER Environment
    Sufijo del stack. Default: dev.

.PARAMETER DomainName
    Dominio raíz. Default: gasparri.com.ar.

.PARAMETER GrafanaSubdomain
    Subdominio para Grafana. Default: grafana → grafana.gasparri.com.ar.

.PARAMETER AdminCidr
    CIDR autorizado a RDS desde el admin (DBeaver). Default: tu IP.

.PARAMETER AlertEmail
    Email para SNS subs de alarmas.

.PARAMETER StartFromPhase
    Reanudar deploys interrumpidos:
      1 = inicio (default)
      2 = desde push de imagen (phase 1 OK)
      3 = desde phase 2 / App Runner deploy (imagen pushed + SQL OK)
      4 = desde associate-custom-domain (App Runner UP)
      5 = desde set de env vars de Grafana (custom domain ACTIVE)

.EXAMPLE
    .\infra\deploy.ps1
    Deploy completo.

.EXAMPLE
    .\infra\deploy.ps1 -StartFromPhase 3
    Reanudar desde el deploy de App Runner.
#>

param(
    [string]$Environment = "dev",
    [string]$DomainName = "tfg.gasparri.com.ar",
    [string]$GrafanaSubdomain = "grafana",
    [string]$AdminCidr = "181.171.27.186/32",
    [string]$AlertEmail = "mauro@gasparri.com.ar",
    [ValidateSet(1, 2, 3, 4, 5)]
    [int]$StartFromPhase = 1
)

$ErrorActionPreference = "Stop"
$STACK_NAME = "people-counter-$Environment"
$REGION = "us-east-1"

# El script vive en infra/, asi que el template y el sql son relativos:
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$TEMPLATE   = Join-Path $SCRIPT_DIR "cloudformation\people-counter.yaml"
$SQL_FILE   = Join-Path $SCRIPT_DIR "sql\bootstrap.sql"

function Get-StackOutput {
    param([string]$Key)
    aws cloudformation describe-stacks --stack-name $STACK_NAME `
        --query "Stacks[0].Outputs[?OutputKey=='$Key'].OutputValue" --output text
}

function Wait-ForUser {
    param([string]$Message)
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host $Message -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Yellow
    Read-Host "Presiona Enter cuando este listo"
}

function Invoke-CfnDeploy {
    param([string]$DeployAppRunner)
    aws cloudformation deploy `
        --stack-name $STACK_NAME `
        --template-file $TEMPLATE `
        --capabilities CAPABILITY_NAMED_IAM `
        --parameter-overrides `
            Environment=$Environment `
            DomainName=$DomainName `
            GrafanaSubdomain=$GrafanaSubdomain `
            AdminCidr=$AdminCidr `
            AlertEmail=$AlertEmail `
            DeployAppRunner=$DeployAppRunner
}

# === Pre-flight ===
Write-Host "Validando template..."
aws cloudformation validate-template --template-body "file://$TEMPLATE" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Template invalido" }

$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
Write-Host "Account: $ACCOUNT_ID  Region: $REGION  Stack: $STACK_NAME"

# === [1/6] Phase 1: deploy core (sin App Runner) ===
if ($StartFromPhase -le 1) {
    Write-Host ""
    Write-Host "[1/6] Phase 1: deploy core (RDS + IoT + Lambda stub + ECR + VPC)" -ForegroundColor Cyan
    Invoke-CfnDeploy "false"
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 deploy fallo" }
    Write-Host "[1/6] OK" -ForegroundColor Green
}

# === [2/6] Push imagen Grafana a ECR ===
if ($StartFromPhase -le 2) {
    Write-Host ""
    Write-Host "[2/6] Pusheando imagen Grafana a ECR..." -ForegroundColor Cyan

    $ECR_URI = Get-StackOutput "GrafanaEcrRepoUri"
    if (-not $ECR_URI) { throw "GrafanaEcrRepoUri no encontrado en outputs" }
    Write-Host "  ECR URI: $ECR_URI"

    # Native commands (docker) escriben warnings a stderr y PS5.1 con
    # $ErrorActionPreference=Stop los trata como excepciones aunque
    # el exit code sea 0. Bajamos temporalmente la preference y validamos
    # explicito con $LASTEXITCODE.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        docker info 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker no esta corriendo o no esta en PATH" }

        $loginPwd = aws ecr get-login-password --region $REGION
        $loginPwd | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker login a ECR fallo" }

        docker pull grafana/grafana:latest
        if ($LASTEXITCODE -ne 0) { throw "Pull de grafana/grafana:latest fallo" }

        docker tag grafana/grafana:latest "${ECR_URI}:latest"
        if ($LASTEXITCODE -ne 0) { throw "Docker tag fallo" }

        docker push "${ECR_URI}:latest"
        if ($LASTEXITCODE -ne 0) { throw "Push a ECR fallo" }
    } finally {
        $ErrorActionPreference = $prevEAP
    }

    Write-Host "[2/6] OK" -ForegroundColor Green
}

# === Bootstrap SQL via docker postgres:16 (psql) ===
if ($StartFromPhase -le 2) {
    Write-Host ""
    Write-Host "  Corriendo bootstrap.sql contra RDS via docker..." -ForegroundColor Cyan

    $RDS_HOST   = Get-StackOutput "RdsEndpoint"
    $RDS_PORT   = Get-StackOutput "RdsPort"
    $SECRET_ARN = Get-StackOutput "RdsMasterSecretArn"

    $secretJson = aws secretsmanager get-secret-value --secret-id $SECRET_ARN --query SecretString --output text
    $secret     = $secretJson | ConvertFrom-Json

    # Docker mount: convertir Windows path a forward slashes
    $sqlDir = (Join-Path $SCRIPT_DIR "sql").Replace('\', '/')

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Array form para evitar problemas de quoting con chars especiales en pwd
        $dockerArgs = @(
            "run", "--rm",
            "-e", "PGPASSWORD=$($secret.password)",
            "-e", "PGSSLMODE=require",
            "-v", "${sqlDir}:/sql:ro",
            "postgres:16",
            "psql",
            "-h", $RDS_HOST,
            "-p", $RDS_PORT,
            "-U", $secret.username,
            "-d", "people_counter",
            "-v", "ON_ERROR_STOP=1",
            "-f", "/sql/bootstrap.sql"
        )
        docker @dockerArgs
        if ($LASTEXITCODE -ne 0) { throw "Bootstrap SQL fallo" }
    } finally {
        $ErrorActionPreference = $prevEAP
    }

    Write-Host "  Bootstrap SQL OK" -ForegroundColor Green
}

# === [3/6] Phase 2: deploy App Runner ===
if ($StartFromPhase -le 3) {
    Write-Host ""
    Write-Host "[3/6] Phase 2: agregando App Runner..." -ForegroundColor Cyan
    Invoke-CfnDeploy "true"
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 deploy fallo" }
    Write-Host "[3/6] OK" -ForegroundColor Green
}

# === [4/6] Associate custom domain ===
if ($StartFromPhase -le 4) {
    Write-Host ""
    Write-Host "[4/6] Asociando custom domain..." -ForegroundColor Cyan

    $SERVICE_ARN = Get-StackOutput "GrafanaServiceArn"
    $FULL_DOMAIN = "$GrafanaSubdomain.$DomainName"

    # Idempotente: skip si ya esta asociado
    $existing = aws apprunner describe-custom-domains --service-arn $SERVICE_ARN `
        --query "CustomDomains[?DomainName=='$FULL_DOMAIN'].DomainName" --output text 2>$null

    if (-not $existing) {
        Write-Host "  Asociando $FULL_DOMAIN..."
        aws apprunner associate-custom-domain `
            --service-arn $SERVICE_ARN `
            --domain-name $FULL_DOMAIN `
            --no-enable-www-subdomain | Out-Null
        Start-Sleep 10   # AWS necesita un momento para generar validation records
    } else {
        Write-Host "  Custom domain ya asociado, refrescando validation records..."
    }

    Write-Host ""
    Write-Host "DNS records a agregar en ${DomainName}:" -ForegroundColor Yellow
    Write-Host ""

    aws apprunner describe-custom-domains --service-arn $SERVICE_ARN `
        --query "CustomDomains[?DomainName=='$FULL_DOMAIN'].CertificateValidationRecords[].{Type:Type, Name:Name, Value:Value, Status:Status}" `
        --output table

    $dnsTarget = aws apprunner describe-custom-domains --service-arn $SERVICE_ARN `
        --query "DNSTarget" --output text

    Write-Host ""
    Write-Host "Y CNAME final para el subdominio:" -ForegroundColor Yellow
    Write-Host "  Nombre:  $GrafanaSubdomain"
    Write-Host "  Tipo:    CNAME"
    Write-Host "  Valor:   $dnsTarget"

    Wait-ForUser "Agrega esos CNAMEs en tu DNS provider (gasparri.com.ar)"

    # Poll until ACTIVE — sigue siendo parte de [4/6] (la fase es "asociar
    # custom domain"; ACTIVE es su definicion de done).
    Write-Host ""
    Write-Host "  Esperando que el custom domain pase a ACTIVE..." -ForegroundColor Cyan

    $maxAttempts = 30
    $attempt = 0
    do {
        $attempt++
        $status = aws apprunner describe-custom-domains --service-arn $SERVICE_ARN `
            --query "CustomDomains[?DomainName=='$FULL_DOMAIN'].Status" --output text
        Write-Host "  [$attempt/$maxAttempts] Status: $status"

        if ($status -eq "ACTIVE") { break }
        if ($status -in @("CREATE_FAILED", "DELETE_FAILED")) {
            throw "Custom domain en estado fallido: $status"
        }

        Start-Sleep 60
    } while ($attempt -lt $maxAttempts)

    if ($status -ne "ACTIVE") {
        Write-Host "Timeout esperando ACTIVE - verifica DNS y volve a correr con -StartFromPhase 4" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "[4/6] OK" -ForegroundColor Green
}

# === [5/6] Setear env vars de Grafana (post-deploy) ===
# El CFN no acepta RuntimeEnvironmentVariables con dynamic references via
# early validation, asi que las seteamos por CLI ahora. Esto triggerea un
# rolling redeploy del service (~2-3 min) que reinicia Grafana apuntando a
# Postgres en RDS (en vez del SQLite default del image).
if ($StartFromPhase -le 5) {
    Write-Host ""
    Write-Host "[5/6] Configurando env vars de Grafana (Postgres backend)..." -ForegroundColor Cyan

    $SERVICE_ARN = Get-StackOutput "GrafanaServiceArn"
    $ECR_URI     = Get-StackOutput "GrafanaEcrRepoUri"
    $RDS_HOST    = Get-StackOutput "RdsEndpoint"
    $RDS_PORT    = Get-StackOutput "RdsPort"
    $SECRET_ARN  = Get-StackOutput "RdsMasterSecretArn"

    $secretJson = aws secretsmanager get-secret-value --secret-id $SECRET_ARN --query SecretString --output text
    $secret     = $secretJson | ConvertFrom-Json

    # AccessRoleArn del service: lo necesitamos para no romperlo en el update.
    $serviceJson     = aws apprunner describe-service --service-arn $SERVICE_ARN --output json
    $service         = $serviceJson | ConvertFrom-Json
    $ACCESS_ROLE_ARN = $service.Service.SourceConfiguration.AuthenticationConfiguration.AccessRoleArn

    $sourceConfig = @{
        ImageRepository = @{
            ImageIdentifier = "${ECR_URI}:latest"
            ImageRepositoryType = "ECR"
            ImageConfiguration = @{
                Port = "3000"
                RuntimeEnvironmentVariables = @{
                    GF_DATABASE_TYPE                = "postgres"
                    GF_DATABASE_HOST                = "${RDS_HOST}:${RDS_PORT}"
                    GF_DATABASE_NAME                = "grafana"
                    GF_DATABASE_USER                = "people_counter"
                    GF_DATABASE_PASSWORD            = $secret.password
                    GF_DATABASE_SSL_MODE            = "require"
                    GF_SERVER_ROOT_URL              = "https://${GrafanaSubdomain}.${DomainName}"
                    GF_SERVER_DOMAIN                = "${GrafanaSubdomain}.${DomainName}"
                    GF_ANALYTICS_REPORTING_ENABLED  = "false"
                    GF_ANALYTICS_CHECK_FOR_UPDATES  = "false"
                    GF_AUTH_ANONYMOUS_ENABLED       = "false"
                    GF_USERS_ALLOW_SIGN_UP          = "false"
                }
            }
        }
        AuthenticationConfiguration = @{
            AccessRoleArn = $ACCESS_ROLE_ARN
        }
        AutoDeploymentsEnabled = $false
    }

    $sourceJson = $sourceConfig | ConvertTo-Json -Depth 10 -Compress

    # AWS CLI no traga BOM. Escribimos UTF-8 sin BOM via .NET (PS 5.1
    # Set-Content -Encoding utf8 mete BOM y rompe el parser).
    $tempFile  = New-TemporaryFile
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($tempFile.FullName, $sourceJson, $utf8NoBom)

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        aws apprunner update-service `
            --service-arn $SERVICE_ARN `
            --source-configuration "file://$($tempFile.FullName)" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "apprunner update-service fallo" }
    } finally {
        $ErrorActionPreference = $prevEAP
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }

    # === [6/6] Poll until RUNNING (post-update) ===
    Write-Host ""
    Write-Host "[6/6] Esperando que App Runner termine el rolling redeploy..." -ForegroundColor Cyan

    $maxAttempts = 20
    $attempt = 0
    do {
        $attempt++
        $svcStatus = aws apprunner describe-service --service-arn $SERVICE_ARN --query "Service.Status" --output text
        Write-Host "  [$attempt/$maxAttempts] Status: $svcStatus"
        if ($svcStatus -eq "RUNNING") { break }
        if ($svcStatus -in @("CREATE_FAILED", "DELETE_FAILED", "PAUSED")) {
            throw "Service en estado fallido: $svcStatus"
        }
        Start-Sleep 30
    } while ($attempt -lt $maxAttempts)

    if ($svcStatus -ne "RUNNING") {
        Write-Host "Timeout esperando RUNNING - revisa CloudWatch del service" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "[6/6] OK" -ForegroundColor Green
}

# === Done ===
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "DEPLOY COMPLETO" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Grafana:        https://$GrafanaSubdomain.$DomainName"
Write-Host "Login inicial:  admin / admin (cambiar al primer login)"
Write-Host ""
Write-Host "Proximos pasos:"
Write-Host "  1. Cambiar password de admin en Grafana"
Write-Host "  2. Deployar el codigo real de Lambda persist_event:"
Write-Host "     .\scripts\deploy_lambda.sh $Environment"
Write-Host "  3. Re-provisionar el device:"
Write-Host "     python scripts\provision.py --thing-name store-pilot-01-cam-01"
