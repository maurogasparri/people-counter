<#
.SYNOPSIS
    Deploy completo del stack people-counter en AWS.

.DESCRIPTION
    Orquesta el deploy en 5 fases del CFN + push de imagen Grafana a ECR +
    cert ACM con validacion DNS + custom domain. Pausa cuando necesita
    accion manual del usuario (correr SQL desde DBeaver es opcional - lo
    cubre el bootstrap automatico - pero pegar CNAMEs en el DNS provider
    externo si es manual).

.PARAMETER Environment
    Sufijo del stack. Default: dev.

.PARAMETER DomainName
    Dominio raiz. Default: tfg.gasparri.com.ar.

.PARAMETER GrafanaSubdomain
    Subdominio para Grafana. Default: grafana -> grafana.tfg.gasparri.com.ar.

.PARAMETER AdminCidr
    CIDR autorizado a RDS desde el admin (DBeaver). Default: tu IP.

.PARAMETER AlertEmail
    Email para SNS subs de alarmas.

.PARAMETER StartFromPhase
    Reanudar deploys interrumpidos:
      1 = inicio (default) - deploy core (RDS + IoT + Lambda + ECR + VPC)
      2 = push imagen ECR + bootstrap SQL
      3 = ACM request-certificate + pause DNS validation + wait ISSUED
      4 = CFN deploy con DeployGrafana=true + cert ARN
      5 = pause CNAME final + verificacion DNS

.EXAMPLE
    .\infra\deploy.ps1
    Deploy completo end-to-end.

.EXAMPLE
    .\infra\deploy.ps1 -StartFromPhase 3
    Reanudar desde el cert ACM (Phase 1+2 OK).
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
$FULL_DOMAIN = "$GrafanaSubdomain.$DomainName"

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
    param(
        [string]$DeployGrafana,
        [string]$CertArn = ""
    )
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
            DeployGrafana=$DeployGrafana `
            GrafanaCertArn=$CertArn
}

# === Pre-flight ===
Write-Host "Validando template..."
aws cloudformation validate-template --template-body "file://$TEMPLATE" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Template invalido" }

$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
Write-Host "Account: $ACCOUNT_ID  Region: $REGION  Stack: $STACK_NAME"
Write-Host "Domain final: https://$FULL_DOMAIN"

# === [1/5] Phase 1: deploy core (sin Grafana) ===
if ($StartFromPhase -le 1) {
    Write-Host ""
    Write-Host "[1/5] Phase 1: deploy core (RDS + IoT + Lambda stub + ECR + VPC)" -ForegroundColor Cyan
    Invoke-CfnDeploy -DeployGrafana "false"
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 deploy fallo" }
    Write-Host "[1/5] OK" -ForegroundColor Green
}

# === [2/5] Push imagen Grafana a ECR + bootstrap SQL ===
if ($StartFromPhase -le 2) {
    Write-Host ""
    Write-Host "[2/5] Pusheando imagen Grafana a ECR..." -ForegroundColor Cyan

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

    Write-Host "  Imagen pushed" -ForegroundColor Green
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

    Write-Host "[2/5] OK" -ForegroundColor Green
}

# === [3/5] ACM cert + DNS validation ===
# Cert se crea FUERA de CFN para que el deploy no bloquee esperando DNS.
# Una vez ISSUED, se pasa el ARN como parametro al CFN en Phase 4. ACM hace
# DNS validation: agrega CNAMEs especificos al DNS provider y ACM checkea
# que existan (re-checkea periodicamente para auto-renew, ergo deben quedar
# permanentes en el DNS provider).
if ($StartFromPhase -le 3) {
    Write-Host ""
    Write-Host "[3/5] Cert ACM para $FULL_DOMAIN..." -ForegroundColor Cyan

    # Idempotente: reusar el cert si ya existe para este FQDN.
    $existingCert = aws acm list-certificates --region $REGION `
        --query "CertificateSummaryList[?DomainName=='$FULL_DOMAIN'].CertificateArn | [0]" `
        --output text 2>$null

    if ($existingCert -and $existingCert -ne "None") {
        $CERT_ARN = $existingCert
        Write-Host "  Cert existente encontrado: $CERT_ARN"
    } else {
        Write-Host "  Solicitando cert nuevo..."
        $CERT_ARN = aws acm request-certificate `
            --region $REGION `
            --domain-name $FULL_DOMAIN `
            --validation-method DNS `
            --query CertificateArn --output text
        if (-not $CERT_ARN) { throw "request-certificate no devolvio ARN" }
        Write-Host "  Cert ARN: $CERT_ARN"
        Start-Sleep 10  # ACM tarda en generar los validation records
    }

    # Status check antes de mostrar validation records
    $certStatus = aws acm describe-certificate --certificate-arn $CERT_ARN `
        --query "Certificate.Status" --output text

    if ($certStatus -eq "ISSUED") {
        Write-Host "  Cert ya esta ISSUED - skip pause de DNS" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "DNS records de validacion a agregar en ${DomainName}:" -ForegroundColor Yellow
        Write-Host "(deben quedar PERMANENTES en el DNS provider — ACM los re-checkea cada renewal)"
        Write-Host ""

        aws acm describe-certificate --certificate-arn $CERT_ARN `
            --query "Certificate.DomainValidationOptions[].ResourceRecord.{Name:Name, Type:Type, Value:Value}" `
            --output table

        Wait-ForUser "Agrega esos CNAMEs en tu DNS provider"

        Write-Host ""
        Write-Host "  Esperando que ACM marque ISSUED..." -ForegroundColor Cyan
        # `aws acm wait certificate-validated` polea cada 60s con timeout 75min.
        aws acm wait certificate-validated --certificate-arn $CERT_ARN
        if ($LASTEXITCODE -ne 0) {
            throw "Cert no llego a ISSUED - verifica que los CNAMEs esten bien y reintenta con -StartFromPhase 3"
        }
    }

    Write-Host "[3/5] OK - cert ISSUED" -ForegroundColor Green

    # Persistir el ARN para Phase 4 si el script se reanuda
    $env:GRAFANA_CERT_ARN = $CERT_ARN
} else {
    # Si arrancamos en Phase >=4, recuperar el cert desde ACM por nombre de domain
    $CERT_ARN = aws acm list-certificates --region $REGION `
        --query "CertificateSummaryList[?DomainName=='$FULL_DOMAIN'].CertificateArn | [0]" `
        --output text
    if (-not $CERT_ARN -or $CERT_ARN -eq "None") {
        throw "No se encontro cert ACM para $FULL_DOMAIN — corre con -StartFromPhase 3"
    }
}

# === [4/5] Phase 2: CFN deploy con Grafana + cert ===
if ($StartFromPhase -le 4) {
    Write-Host ""
    Write-Host "[4/5] Phase 2: deploy ECS Fargate + ALB + Grafana..." -ForegroundColor Cyan
    Write-Host "  (CFN espera a que el service estabilice, ~5-8 min)"
    Invoke-CfnDeploy -DeployGrafana "true" -CertArn $CERT_ARN
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 deploy fallo" }
    Write-Host "[4/5] OK" -ForegroundColor Green
}

# === [5/5] CNAME final ===
if ($StartFromPhase -le 5) {
    Write-Host ""
    Write-Host "[5/5] CNAME final $FULL_DOMAIN -> ALB..." -ForegroundColor Cyan

    $ALB_DNS = Get-StackOutput "GrafanaAlbDnsName"
    if (-not $ALB_DNS) { throw "GrafanaAlbDnsName no encontrado en outputs" }

    Write-Host ""
    Write-Host "Agregar al DNS provider:" -ForegroundColor Yellow
    Write-Host "  Nombre:  $GrafanaSubdomain"
    Write-Host "  Tipo:    CNAME"
    Write-Host "  Valor:   $ALB_DNS"
    Write-Host ""

    Wait-ForUser "Agrega el CNAME en tu DNS provider"

    # Verificacion de resolucion DNS (best-effort, no falla si TTL todavia no propago)
    Write-Host ""
    Write-Host "  Verificando que $FULL_DOMAIN resuelva al ALB..." -ForegroundColor Cyan
    $maxAttempts = 6
    $attempt = 0
    do {
        $attempt++
        try {
            $resolved = (Resolve-DnsName -Name $FULL_DOMAIN -Type CNAME -ErrorAction Stop -DnsOnly).NameHost
            if ($resolved -and $resolved -like "*$ALB_DNS*") {
                Write-Host "  Resuelto: $FULL_DOMAIN -> $resolved" -ForegroundColor Green
                break
            }
            Write-Host "  [$attempt/$maxAttempts] Aun no propago (resolved: $resolved)"
        } catch {
            Write-Host "  [$attempt/$maxAttempts] Aun no propago (sin resolucion)"
        }
        if ($attempt -lt $maxAttempts) { Start-Sleep 30 }
    } while ($attempt -lt $maxAttempts)

    Write-Host "[5/5] OK" -ForegroundColor Green
}

# === Done ===
$GRAFANA_URL = Get-StackOutput "GrafanaUrl"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "DEPLOY COMPLETO" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Grafana:        $GRAFANA_URL"
Write-Host "Login inicial:  admin / admin (cambiar al primer login)"
Write-Host ""
Write-Host "Proximos pasos:"
Write-Host "  1. Cambiar password de admin en Grafana"
Write-Host "  2. Deployar el codigo real de Lambda persist_event:"
Write-Host "     .\scripts\deploy_lambda.sh $Environment"
Write-Host "  3. Re-provisionar el device:"
Write-Host "     python scripts\provision.py --thing-name store-pilot-01-cam-01"
