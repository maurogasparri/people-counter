<#
.SYNOPSIS
    Teardown del stack people-counter en AWS para un redeploy limpio.

.DESCRIPTION
    Borra el stack CloudFormation `people-counter-<env>` y limpia los
    recursos que sobreviven al delete y romperian/ensuciarian un redeploy.

    Sirve para cualquier version del stack que este viva (App Runner viejo,
    ECS Express, o Fargate+ALB) - todas usaron el mismo stack name, asi que
    `delete-stack` borra lo que sea que este deployado. El script muestra los
    recursos del stack ANTES de borrar para que confirmes que es lo que
    esperabas.

    Que hace el delete del stack automaticamente (por el template):
      - RDS: toma un SNAPSHOT FINAL (DeletionPolicy: Snapshot) y borra la
        instancia. Los datos quedan en el snapshot.
      - ECR: se vacia solo (EmptyOnDelete: true) y se borra.
      - App Runner Service / ECS+ALB / Lambda / IoT Rules: se borran.

    Que NO borra el delete del stack y limpia este script:
      - SecretsManager `people-counter/<env>/rds-master`: el delete del stack
        lo deja en RECOVERY WINDOW (7-30 dias). Un redeploy con el mismo
        nombre CHOCA. Lo force-delete-amos sin recovery.
      - Log groups que Lambda/ECS pudieron recrear post-delete.
      - (opcional) snapshot final de RDS  -> -DeleteSnapshot  (PIERDE DATOS).
      - (opcional) cert ACM out-of-CFN     -> -DeleteCert     (re-validacion
        DNS en el proximo deploy; por default se CONSERVA).
      - (best-effort) App Runner service orphan + su custom domain association.

    Que queda para vos (manual, DNS provider externo):
      - El CNAME grafana->ALB/AppRunner queda STALE (el redeploy crea otro
        endpoint). Hay que actualizarlo despues del nuevo deploy.
      - Los CNAMEs de validacion ACM: dejarlos si conservas el cert.

.PARAMETER Environment
    Sufijo del stack. Default: dev  -> stack people-counter-dev.

.PARAMETER Region
    Default: us-east-1.

.PARAMETER DomainName / GrafanaSubdomain
    Para localizar el cert ACM si pasas -DeleteCert. Defaults igual que deploy.ps1.

.PARAMETER DeleteSnapshot
    Tambien borra el snapshot final de RDS. DESTRUCTIVO: pierde los datos.

.PARAMETER DeleteCert
    Tambien borra el cert ACM del FQDN de Grafana. Por default se conserva
    (evita re-validar DNS en el proximo deploy).

.PARAMETER Force
    Skip de la confirmacion interactiva. Usar con cuidado.

.EXAMPLE
    .\infra\teardown.ps1
    Borra el stack people-counter-dev, conserva snapshot RDS y cert ACM.

.EXAMPLE
    .\infra\teardown.ps1 -DeleteSnapshot -DeleteCert -Force
    Teardown total sin prompts (pierde datos y cert).
#>

param(
    [string]$Environment = "dev",
    [string]$Region = "us-east-1",
    [string]$DomainName = "tfg.gasparri.com.ar",
    [string]$GrafanaSubdomain = "grafana",
    [switch]$DeleteSnapshot,
    [switch]$DeleteCert,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$STACK_NAME = "people-counter-$Environment"
# Los dos secrets de Secrets Manager que el CFN crea: ambos quedan en recovery
# window (7-30 dias) al delete-stack y chocan en un redeploy con el mismo nombre,
# asi que se force-deletean sin recovery.
$SECRET_IDS = @(
    "people-counter/$Environment/rds-master",   # RDS master (IAM auth + admin)
    "people-counter-ses-smtp-$Environment"        # SES SMTP creds (Grafana alerting)
)
$RDS_ID     = "people-counter-$Environment"
$FULL_DOMAIN = "$GrafanaSubdomain.$DomainName"
$API_DOMAIN  = "api.$DomainName"

function Test-StackExists {
    aws cloudformation describe-stacks --stack-name $STACK_NAME --region $Region 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# === Pre-flight ===
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0) { throw "No hay credenciales AWS validas (aws sts get-caller-identity fallo)" }
Write-Host "Account: $ACCOUNT_ID  Region: $Region  Stack: $STACK_NAME" -ForegroundColor Cyan

$stackExists = Test-StackExists

if ($stackExists) {
    Write-Host ""
    Write-Host "Recursos del stack que se van a BORRAR:" -ForegroundColor Yellow
    aws cloudformation list-stack-resources --stack-name $STACK_NAME --region $Region `
        --query "StackResourceSummaries[].{Type:ResourceType, Logical:LogicalResourceId, Status:ResourceStatus}" `
        --output table
} else {
    Write-Host "El stack $STACK_NAME no existe (ya borrado o nunca creado)." -ForegroundColor Yellow
    Write-Host "Sigo igual con la limpieza de leftovers out-of-CFN."
}

Write-Host ""
Write-Host "Tambien se va a:" -ForegroundColor Yellow
Write-Host "  - force-delete de los secrets (sin recovery window):"
foreach ($sid in $SECRET_IDS) { Write-Host "      $sid" }
if ($DeleteSnapshot) {
    Write-Host "  - BORRAR el snapshot final de RDS ($RDS_ID) -> SE PIERDEN LOS DATOS" -ForegroundColor Red
} else {
    Write-Host "  - CONSERVAR el snapshot final de RDS (usa -DeleteSnapshot para borrarlo)"
}
if ($DeleteCert) {
    Write-Host "  - BORRAR los certs ACM de $FULL_DOMAIN y $API_DOMAIN (requieren re-validacion DNS al redeploy)" -ForegroundColor Red
} else {
    Write-Host "  - CONSERVAR los certs ACM de $FULL_DOMAIN y $API_DOMAIN (usa -DeleteCert para borrarlos)"
}

# === Confirmacion ===
if (-not $Force) {
    Write-Host ""
    $confirm = Read-Host "Escribi DELETE para confirmar el teardown"
    if ($confirm -ne "DELETE") {
        Write-Host "Abortado (no se escribio DELETE)." -ForegroundColor Yellow
        exit 0
    }
}

# === [1/5] Delete del stack ===
if ($stackExists) {
    Write-Host ""
    Write-Host "[1/5] Borrando stack $STACK_NAME (RDS toma snapshot final, ~5-10 min)..." -ForegroundColor Cyan
    aws cloudformation delete-stack --stack-name $STACK_NAME --region $Region
    if ($LASTEXITCODE -ne 0) { throw "delete-stack fallo" }
    Write-Host "  Esperando stack-delete-complete..."
    aws cloudformation wait stack-delete-complete --stack-name $STACK_NAME --region $Region
    if ($LASTEXITCODE -ne 0) {
        throw "El stack no termino de borrarse. Revisa en la consola CFN si quedo un recurso con DELETE_FAILED (tipico: ENI de Lambda/ECS o dependencia de VPC)."
    }
    Write-Host "[1/5] Stack borrado" -ForegroundColor Green
} else {
    Write-Host "[1/5] Skip (stack inexistente)" -ForegroundColor Green
}

# === [2/5] Force-delete del secret ===
Write-Host ""
Write-Host "[2/5] Force-delete de los secrets (sin recovery window)..." -ForegroundColor Cyan
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
foreach ($sid in $SECRET_IDS) {
    aws secretsmanager delete-secret --secret-id $sid --region $Region `
        --force-delete-without-recovery 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  eliminado: $sid" -ForegroundColor Green
    } else {
        Write-Host "  no existia o ya borrado: $sid (ok)" -ForegroundColor Green
    }
}
$ErrorActionPreference = $prevEAP
Write-Host "[2/5] Secrets limpiados" -ForegroundColor Green

# === [3/5] App Runner leftover (orphan del deploy viejo) ===
Write-Host ""
Write-Host "[3/5] Buscando App Runner service orphan (deploy viejo)..." -ForegroundColor Cyan
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$arSvcArn = aws apprunner list-services --region $Region `
    --query "ServiceSummaryList[?ServiceName=='people-counter-grafana-$Environment'].ServiceArn | [0]" `
    --output text 2>$null
if ($arSvcArn -and $arSvcArn -ne "None") {
    Write-Host "  Encontrado App Runner service orphan: $arSvcArn" -ForegroundColor Yellow
    Write-Host "  Borrando (la custom domain association se remueve con el service)..."
    aws apprunner delete-service --service-arn $arSvcArn --region $Region 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[3/5] App Runner service borrado" -ForegroundColor Green
    } else {
        Write-Host "[3/5] No se pudo borrar el App Runner service - revisalo en consola" -ForegroundColor Yellow
    }
} else {
    Write-Host "[3/5] Sin App Runner service orphan (ok)" -ForegroundColor Green
}
$ErrorActionPreference = $prevEAP

# === [4/5] Log groups leftover ===
Write-Host ""
Write-Host "[4/5] Limpiando log groups leftover..." -ForegroundColor Cyan
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
# Incluye los log groups que servicios crean por su cuenta (NO CFN-managed,
# por eso sobreviven al delete del stack): App Runner (/aws/apprunner) y RDS
# (/aws/rds/instance) ademas de los de Lambda/ECS.
$logPrefixes = @(
    "/aws/lambda/people-counter",
    "/ecs/people-counter",
    "/aws/apprunner/people-counter",
    "/aws/rds/instance/people-counter"
)
foreach ($prefix in $logPrefixes) {
    $groups = aws logs describe-log-groups --region $Region `
        --log-group-name-prefix $prefix `
        --query "logGroups[?contains(logGroupName, '-$Environment')].logGroupName" `
        --output text 2>$null
    if ($groups -and $groups -ne "None") {
        foreach ($g in ($groups -split "\s+")) {
            if ($g) {
                aws logs delete-log-group --log-group-name $g --region $Region 2>$null | Out-Null
                Write-Host "  borrado: $g"
            }
        }
    }
}
Write-Host "[4/5] Log groups limpiados" -ForegroundColor Green
$ErrorActionPreference = $prevEAP

# === [5/5] Opcionales: snapshot RDS + cert ACM ===
Write-Host ""
Write-Host "[5/5] Recursos opcionales..." -ForegroundColor Cyan
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"

if ($DeleteSnapshot) {
    $snaps = aws rds describe-db-snapshots --region $Region `
        --db-instance-identifier $RDS_ID --snapshot-type manual `
        --query "DBSnapshots[].DBSnapshotIdentifier" --output text 2>$null
    if ($snaps -and $snaps -ne "None") {
        foreach ($s in ($snaps -split "\s+")) {
            if ($s) {
                aws rds delete-db-snapshot --db-snapshot-identifier $s --region $Region 2>$null | Out-Null
                Write-Host "  snapshot borrado: $s"
            }
        }
    } else {
        Write-Host "  sin snapshots manuales para $RDS_ID"
    }
} else {
    Write-Host "  snapshot RDS conservado (sin -DeleteSnapshot)"
}

if ($DeleteCert) {
    foreach ($certFqdn in @($FULL_DOMAIN, $API_DOMAIN)) {
        $certArn = aws acm list-certificates --region $Region `
            --query "CertificateSummaryList[?DomainName=='$certFqdn'].CertificateArn | [0]" `
            --output text 2>$null
        if ($certArn -and $certArn -ne "None") {
            aws acm delete-certificate --certificate-arn $certArn --region $Region 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  cert ACM borrado ($certFqdn): $certArn"
            } else {
                Write-Host "  no se pudo borrar el cert de $certFqdn (puede estar en uso por otro recurso)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  sin cert ACM para $certFqdn"
        }
    }
} else {
    Write-Host "  certs ACM conservados (sin -DeleteCert)"
}
$ErrorActionPreference = $prevEAP

# === Done ===
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "TEARDOWN COMPLETO" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Pendiente manual en tu DNS provider:" -ForegroundColor Yellow
Write-Host "  - El CNAME $FULL_DOMAIN -> (ALB viejo) quedo STALE."
Write-Host "    Lo actualizas DESPUES del nuevo deploy (apunta a otro endpoint)."
Write-Host "  - El CNAME $API_DOMAIN -> (API Gateway viejo) tambien quedo STALE."
Write-Host "  - Los 3 CNAMEs DKIM de SES quedan STALE si recreas la identity en el redeploy."
if (-not $DeleteCert) {
    Write-Host "  - Los CNAMEs de validacion ACM: DEJALOS (los certs se conservaron)."
}
Write-Host ""
Write-Host "Para el deploy nuevo:" -ForegroundColor Cyan
Write-Host "  .\infra\deploy.ps1"
