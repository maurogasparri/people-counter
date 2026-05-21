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

.PARAMETER AlertEmail
    Email para SNS subs de alarmas.

.PARAMETER StartFromPhase
    Reanudar deploys interrumpidos:
      1 = inicio (default) - deploy core (RDS + IoT + Lambda + ECR + VPC)
      2 = push imagen ECR + bootstrap SQL
      3 = ACM certs (grafana + api) + pause DNS validation + wait ISSUED
      4 = CFN deploy con DeployGrafana=true + ambos cert ARNs
      5 = pause CNAMEs finales (grafana -> ALB, api -> API GW) + verificacion
      6 = crear datasource Postgres en Grafana via API (admin/admin)

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
    [string]$AlertEmail = "mauro@gasparri.com.ar",
    [ValidateSet(1, 2, 3, 4, 5, 6)]
    [int]$StartFromPhase = 1
)

$ErrorActionPreference = "Stop"
$STACK_NAME = "people-counter-$Environment"
$REGION = "us-east-1"
$FULL_DOMAIN = "$GrafanaSubdomain.$DomainName"
$API_DOMAIN  = "api.$DomainName"

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
        [string]$CertArn = "",
        [string]$ApiCertArn = ""
    )
    aws cloudformation deploy `
        --stack-name $STACK_NAME `
        --template-file $TEMPLATE `
        --capabilities CAPABILITY_NAMED_IAM `
        --parameter-overrides `
            Environment=$Environment `
            DomainName=$DomainName `
            GrafanaSubdomain=$GrafanaSubdomain `
            AlertEmail=$AlertEmail `
            DeployGrafana=$DeployGrafana `
            GrafanaCertArn=$CertArn `
            ApiCertArn=$ApiCertArn
}

# Solicita-o-reusa un cert ACM para $Fqdn, pausa para la validacion DNS si
# hace falta, espera ISSUED, y devuelve el ARN. Cuidado con el pipeline:
# solo $arn debe salir de la funcion (los displays van a Out-Host/Out-Null).
function Ensure-Cert {
    param([string]$Fqdn)
    Write-Host "  Cert ACM para $Fqdn..." -ForegroundColor Cyan
    $arn = aws acm list-certificates --region $REGION `
        --query "CertificateSummaryList[?DomainName=='$Fqdn'].CertificateArn | [0]" `
        --output text 2>$null
    if ($arn -and $arn -ne "None") {
        Write-Host "    Cert existente: $arn"
    } else {
        $arn = aws acm request-certificate --region $REGION --domain-name $Fqdn `
            --validation-method DNS --query CertificateArn --output text
        if (-not $arn) { throw "request-certificate fallo para $Fqdn" }
        Write-Host "    Cert nuevo: $arn"
        Start-Sleep 10  # ACM tarda en generar los validation records
    }
    $status = aws acm describe-certificate --certificate-arn $arn `
        --query "Certificate.Status" --output text
    if ($status -eq "ISSUED") {
        Write-Host "    ya ISSUED" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "DNS de validacion para ${Fqdn} (PERMANENTE - ACM lo re-checkea cada renewal):" -ForegroundColor Yellow
        aws acm describe-certificate --certificate-arn $arn `
            --query "Certificate.DomainValidationOptions[].ResourceRecord.{Name:Name,Type:Type,Value:Value}" `
            --output table | Out-Host
        Wait-ForUser "Agrega el CNAME de validacion de $Fqdn en tu DNS" | Out-Null
        Write-Host "    esperando ISSUED..." -ForegroundColor Cyan
        aws acm wait certificate-validated --certificate-arn $arn | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Cert de $Fqdn no llego a ISSUED (revisa el CNAME, reintenta -StartFromPhase 3)" }
    }
    return $arn
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
        # --password directo (no --password-stdin): el pipe de PowerShell a
        # stdin corrompe el token (encoding/newline) y ECR responde 400.
        docker login --username AWS --password $loginPwd "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com" 2>&1 | Out-Null
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

# === [3/6] ACM certs (grafana + api) + DNS validation ===
# Los certs se crean FUERA de CFN para que el deploy no bloquee esperando la
# validacion DNS. Una vez ISSUED, los ARNs entran como parametros al CFN en
# Phase 4. Ensure-Cert pausa pidiendo el CNAME de validacion si hace falta.
if ($StartFromPhase -le 3) {
    Write-Host ""
    Write-Host "[3/6] Certs ACM (grafana + api)..." -ForegroundColor Cyan
    $CERT_ARN     = Ensure-Cert $FULL_DOMAIN
    $API_CERT_ARN = Ensure-Cert $API_DOMAIN
    Write-Host "[3/6] OK - ambos certs ISSUED" -ForegroundColor Green
} else {
    # Reanudando en Phase >=4: recuperar ambos ARNs por nombre de dominio.
    $CERT_ARN = aws acm list-certificates --region $REGION `
        --query "CertificateSummaryList[?DomainName=='$FULL_DOMAIN'].CertificateArn | [0]" --output text
    if (-not $CERT_ARN -or $CERT_ARN -eq "None") { throw "No se encontro cert para $FULL_DOMAIN - corre -StartFromPhase 3" }
    $API_CERT_ARN = aws acm list-certificates --region $REGION `
        --query "CertificateSummaryList[?DomainName=='$API_DOMAIN'].CertificateArn | [0]" --output text
    if (-not $API_CERT_ARN -or $API_CERT_ARN -eq "None") { throw "No se encontro cert para $API_DOMAIN - corre -StartFromPhase 3" }
}

# === [4/6] CFN deploy con Grafana + ambos certs ===
if ($StartFromPhase -le 4) {
    Write-Host ""
    Write-Host "[4/6] deploy ECS Fargate + ALB + Grafana + custom domain API..." -ForegroundColor Cyan
    Write-Host "  (CFN espera a que el service estabilice, ~5-8 min)"
    # ECS service-linked role: en cuentas que nunca usaron ECS no existe, y
    # CreateCluster falla con "Unable to assume the service linked role".
    # Crearlo es idempotente; si ya existe AWS responde "has been taken".
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com 2>&1 | Out-Null
    $ErrorActionPreference = $prevEAP
    Invoke-CfnDeploy -DeployGrafana "true" -CertArn $CERT_ARN -ApiCertArn $API_CERT_ARN
    if ($LASTEXITCODE -ne 0) { throw "Phase 4 deploy fallo" }
    Write-Host "[4/6] OK" -ForegroundColor Green

    # --- Codigo real de la Lambda persist_event ---
    # El template crea la Lambda con un stub inline (ImportModuleError al
    # invocarla). El deploy del codigo real va ACA, al final de Phase 4, porque
    # CADA stack deploy (Phase 1 y 4) resetea la Lambda al stub inline -> hay
    # que re-deployar el codigo justo despues del ultimo stack deploy. Antes era
    # un paso manual (scripts/deploy_lambda.sh) que se olvidaba en cada redeploy.
    Write-Host "  Deployando codigo real de la Lambda persist_event..." -ForegroundColor Cyan
    $lambdaFn  = "people-counter-persist-event-$Environment"
    $lambdaSrc = Join-Path $SCRIPT_DIR "..\src\cloud\persist_event.py"
    $buildDir  = Join-Path $env:TEMP "pc-lambda-build"
    $zipPath   = Join-Path $env:TEMP "pc-lambda.zip"
    Remove-Item -Recurse -Force $buildDir, $zipPath -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $buildDir | Out-Null
    # psycopg (binary wheel) target el runtime de Lambda: manylinux2014 x86_64,
    # py3.13. Usamos `py -m pip` porque `pip` suelto no siempre esta en PATH
    # (git-bash/Windows). stderr de pip (notices) va a consola, sin redirigir.
    py -m pip install --platform manylinux2014_x86_64 --target $buildDir `
        --implementation cp --python-version 3.13 --only-binary=:all: --upgrade `
        "psycopg[binary]==3.2.*" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pip install de psycopg para la Lambda fallo" }
    Copy-Item $lambdaSrc (Join-Path $buildDir "persist_event.py") -Force
    # Zip via zipfile de Python: garantiza arcnames con '/' (Compress-Archive de
    # PS 5.1 usa '\' y Lambda no encuentra el modulo -> ImportModuleError).
    py -c "import zipfile,os,sys; b=sys.argv[1]; z=zipfile.ZipFile(sys.argv[2],'w',zipfile.ZIP_DEFLATED); [z.write(os.path.join(r,f), os.path.relpath(os.path.join(r,f),b).replace(os.sep,'/')) for r,_,fs in os.walk(b) for f in fs]; z.close()" $buildDir $zipPath
    if ($LASTEXITCODE -ne 0) { throw "Empaquetado (zip) de la Lambda fallo" }
    $sz = aws lambda update-function-code --function-name $lambdaFn `
        --zip-file "fileb://$zipPath" --query 'CodeSize' --output text
    if ($LASTEXITCODE -ne 0) { throw "update-function-code de la Lambda fallo" }
    Write-Host "  Lambda persist_event deployada (CodeSize: $sz bytes)" -ForegroundColor Green
}

# === [5/6] CNAMEs finales (grafana -> ALB, api -> API GW domain) ===
if ($StartFromPhase -le 5) {
    Write-Host ""
    Write-Host "[5/6] CNAMEs finales..." -ForegroundColor Cyan

    $ALB_DNS = Get-StackOutput "GrafanaAlbDnsName"
    if (-not $ALB_DNS) { throw "GrafanaAlbDnsName no encontrado en outputs" }
    $API_TARGET = Get-StackOutput "IngestPosDomainTarget"
    if (-not $API_TARGET) { throw "IngestPosDomainTarget no encontrado en outputs" }

    $zonePrefix = $DomainName.Split('.')[0]   # ej. tfg.gasparri.com.ar -> tfg
    Write-Host ""
    Write-Host "Agregar al DNS provider (2 CNAMEs):" -ForegroundColor Yellow
    Write-Host "  1) $FULL_DOMAIN   CNAME  $ALB_DNS       (relativo: $GrafanaSubdomain.$zonePrefix)"
    Write-Host "  2) $API_DOMAIN          CNAME  $API_TARGET   (relativo: api.$zonePrefix)"
    Write-Host ""

    Wait-ForUser "Agrega ambos CNAMEs en tu DNS provider"

    # Verificacion de resolucion (best-effort, no falla si el TTL no propago)
    foreach ($pair in @(@($FULL_DOMAIN, $ALB_DNS), @($API_DOMAIN, $API_TARGET))) {
        $fqdn = $pair[0]; $tgt = $pair[1]
        Write-Host "  Verificando $fqdn -> $tgt ..." -ForegroundColor Cyan
        $attempt = 0
        do {
            $attempt++
            try {
                $resolved = (Resolve-DnsName -Name $fqdn -Type CNAME -ErrorAction Stop -DnsOnly).NameHost
                if ($resolved -and $resolved -like "*$tgt*") { Write-Host "    OK: $resolved" -ForegroundColor Green; break }
                Write-Host "    [$attempt/6] aun no propago (resolved: $resolved)"
            } catch { Write-Host "    [$attempt/6] aun no propago" }
            if ($attempt -lt 6) { Start-Sleep 30 }
        } while ($attempt -lt 6)
    }
    Write-Host "[5/6] OK" -ForegroundColor Green
}

# === [6/6] Datasource Postgres en Grafana via API ===
# Se hace despues del CNAME (Phase 5) porque pega a https://<grafana fqdn>.
# Usa admin/admin: funciona en un Grafana fresco antes del primer login. Si
# ya cambiaste el password, da 401 y se skipea (agregalo manual en la UI).
if ($StartFromPhase -le 6) {
    Write-Host ""
    Write-Host "[6/6] Datasource Postgres en Grafana..." -ForegroundColor Cyan

    $rdsHost   = Get-StackOutput "RdsEndpoint"
    $secretArn = Get-StackOutput "RdsMasterSecretArn"
    $sec = (aws secretsmanager get-secret-value --secret-id $secretArn --query SecretString --output text) | ConvertFrom-Json
    $cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:admin"))
    $headers = @{ Authorization = "Basic $cred" }
    $gUrl = "https://$FULL_DOMAIN"

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        Invoke-RestMethod -Uri "$gUrl/api/user" -Headers $headers -TimeoutSec 15 | Out-Null
        $authOk = $true
    } catch {
        $authOk = $false
        Write-Host "  admin/admin no autentico (ya cambiaste el password?). Skip." -ForegroundColor Yellow
        Write-Host "  Agregalo manual: Connections > Data sources > PostgreSQL"
        Write-Host "    host $($rdsHost):5432  db people_counter  user people_counter  SSL=require"
    }

    if ($authOk) {
        $body = @{
            name = "people-counter"; type = "postgres"; access = "proxy"
            url = "$($rdsHost):5432"; user = "people_counter"; isDefault = $true; database = "people_counter"
            jsonData = @{ database = "people_counter"; sslmode = "require"; postgresVersion = 1600; timescaledb = $false }
            secureJsonData = @{ password = $sec.password }
        } | ConvertTo-Json -Depth 5
        try {
            $ds = Invoke-RestMethod -Uri "$gUrl/api/datasources" -Method Post -Headers $headers -ContentType "application/json" -Body $body -TimeoutSec 20
            $h = Invoke-RestMethod -Uri "$gUrl/api/datasources/uid/$($ds.datasource.uid)/health" -Headers $headers -TimeoutSec 25
            Write-Host "  datasource creado - health: $($h.status) ($($h.message))" -ForegroundColor Green
        } catch {
            $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
            if ($code -eq 409) { Write-Host "  datasource ya existia (ok)" -ForegroundColor Green }
            else { Write-Host "  no se pudo crear el datasource (status $code) - agregalo manual en la UI" -ForegroundColor Yellow }
        }
    }
    $ErrorActionPreference = $prevEAP
    Write-Host "[6/6] OK" -ForegroundColor Green
}

# === Done ===
$GRAFANA_URL = Get-StackOutput "GrafanaUrl"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "DEPLOY COMPLETO" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Grafana:   $GRAFANA_URL  (login inicial admin / admin)"
Write-Host "POS API:   https://$API_DOMAIN/pos/transactions  (POST, firma SigV4 / IAM)"
Write-Host ""
$IOT_EP = aws iot describe-endpoint --endpoint-type iot:Data-ATS --query endpointAddress --output text
Write-Host "Proximos pasos:"
Write-Host "  1. Cambiar password de admin en Grafana"
Write-Host "  2. Re-provisionar el device (crea thing + cert + config + deploy SSH):"
Write-Host "     python scripts\provision.py create --device-id store-pilot-01-cam-01 --store-id store-pilot-01 --endpoint $IOT_EP --policy-name people-counter-device-policy-$Environment"
Write-Host "  (El codigo real de la Lambda persist_event ya se deploya solo en Phase 4, no es paso manual.)"
