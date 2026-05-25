<#
.SYNOPSIS
    Importa todos los dashboards JSON de infra/grafana/dashboards/ a Grafana
    via API. Idempotente: si el dashboard ya existe (por uid), lo sobrescribe.

.DESCRIPTION
    Flow:
      1. Obtiene el UID de la datasource ``people-counter`` (Postgres) via API.
      2. Para cada *.json en infra/grafana/dashboards/:
         a. Sustituye el placeholder ``__DS_UID__`` por el UID real.
         b. Envuelve en {"dashboard": ..., "overwrite": true}.
         c. POST a /api/dashboards/db.
      3. Reporta éxito o falla por archivo.

    Auth (en orden de preferencia):
      1. ``$env:GRAFANA_API_TOKEN`` — Bearer token de un service account
         (recomendado en producción; rol Editor alcanza). Crear via UI:
         Administration → Service accounts → Add → Add token.
      2. ``$env:GRAFANA_ADMIN_PASSWORD`` + ``$env:GRAFANA_ADMIN_USER``
         opcional (default admin) — basic auth con usuario humano. Útil
         solo para bootstrap inicial. Si el admin cambió de password
         después del deploy, hay que pasarlo por env.
      3. Fallback ``admin/admin`` (solo Grafana recién deployado).

.PARAMETER GrafanaUrl
    URL base de Grafana. Default: https://grafana.tfg.gasparri.com.ar

.PARAMETER DashboardsDir
    Directorio con los JSON a importar. Default: infra/grafana/dashboards (relativo a este script).

.EXAMPLE
    .\infra\grafana\import_dashboards.ps1
    Importa todos los dashboards al stack productivo.

.EXAMPLE
    $env:GRAFANA_ADMIN_PASSWORD = "<nuevo password>"
    .\infra\grafana\import_dashboards.ps1
    Idem con password custom.
#>

param(
    [string]$GrafanaUrl = "https://grafana.tfg.gasparri.com.ar",
    [string]$DashboardsDir = ""
)

$ErrorActionPreference = "Stop"

# Resolver path relativo al script si no se pasa explícito.
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $DashboardsDir) {
    $DashboardsDir = Join-Path $SCRIPT_DIR "dashboards"
}
if (-not (Test-Path $DashboardsDir)) {
    throw "No existe el directorio de dashboards: $DashboardsDir"
}

# Auth: preferir service account token sobre basic auth.
if ($env:GRAFANA_API_TOKEN) {
    $authHeader = @{ Authorization = "Bearer $($env:GRAFANA_API_TOKEN)" }
    $authMode = "Bearer (service account)"
} else {
    $adminUser = if ($env:GRAFANA_ADMIN_USER) { $env:GRAFANA_ADMIN_USER } else { "admin" }
    $adminPass = if ($env:GRAFANA_ADMIN_PASSWORD) { $env:GRAFANA_ADMIN_PASSWORD } else { "admin" }
    $credBytes = [Text.Encoding]::ASCII.GetBytes("${adminUser}:${adminPass}")
    $authHeader = @{ Authorization = "Basic " + [Convert]::ToBase64String($credBytes) }
    $authMode = "Basic ($adminUser)"
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Import de dashboards Grafana"  -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "URL:          $GrafanaUrl"
Write-Host "Dashboards:   $DashboardsDir"
Write-Host "Auth:         $authMode"
Write-Host ""

# --- Paso 1: obtener UID de la datasource ``people-counter`` ---
Write-Host "[1/3] Obteniendo UID de la datasource people-counter..." -ForegroundColor Cyan
try {
    $ds = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/name/people-counter" `
        -Headers $authHeader -TimeoutSec 15
} catch {
    $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
    if ($code -eq 401) {
        throw "Auth fallo (401). Opciones: (1) exportar GRAFANA_API_TOKEN con un service account token; (2) exportar GRAFANA_ADMIN_PASSWORD con el password actual del admin."
    } elseif ($code -eq 403) {
        throw "Auth OK pero sin permisos (403). El service account necesita rol Editor o Admin."
    } elseif ($code -eq 404) {
        throw "Datasource 'people-counter' no encontrada. Correr deploy.ps1 Fase 6 para crearla primero."
    } else {
        throw "Error consultando la datasource (HTTP $code): $($_.Exception.Message)"
    }
}
$dsUid = $ds.uid
Write-Host "  UID: $dsUid" -ForegroundColor Green

# --- Paso 2: importar cada JSON ---
Write-Host ""
Write-Host "[2/3] Importando dashboards..." -ForegroundColor Cyan

$jsonFiles = Get-ChildItem -Path $DashboardsDir -Filter "*.json" | Sort-Object Name
if (-not $jsonFiles) {
    Write-Host "  No hay archivos JSON en $DashboardsDir" -ForegroundColor Yellow
    exit 0
}

$ok = 0
$failed = 0
foreach ($file in $jsonFiles) {
    Write-Host "  - $($file.Name)..." -NoNewline
    try {
        # Leer JSON + sustituir placeholder de la datasource.
        $rawJson = Get-Content $file.FullName -Raw -Encoding UTF8
        $rawJson = $rawJson -replace "__DS_UID__", $dsUid

        # Validar que parsea como JSON (catch errores de sintaxis pre-POST).
        $dashObj = $rawJson | ConvertFrom-Json

        # Wrappear en el envelope que espera la API + escribir a archivo temporal
        # (POST como string con caracteres no-ASCII se corrompe via Invoke-RestMethod
        # en PS 5.1; usar -InFile con UTF8 garantiza encoding correcto).
        $envelope = @{
            dashboard = $dashObj
            overwrite = $true
            message   = "Import via import_dashboards.ps1"
            folderUid = ""
        } | ConvertTo-Json -Depth 100 -Compress

        $tmpFile = Join-Path $env:TEMP "grafana-dash-$($file.BaseName).json"
        [IO.File]::WriteAllBytes($tmpFile, [Text.Encoding]::UTF8.GetBytes($envelope))

        $resp = Invoke-RestMethod -Uri "$GrafanaUrl/api/dashboards/db" `
            -Method Post `
            -Headers $authHeader `
            -ContentType "application/json; charset=utf-8" `
            -InFile $tmpFile `
            -TimeoutSec 30

        Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue

        if ($resp.status -eq "success") {
            Write-Host " OK (uid=$($resp.uid), version=$($resp.version), slug=$($resp.slug))" -ForegroundColor Green
            $ok++
        } else {
            Write-Host " FALLA: respuesta inesperada: $($resp | ConvertTo-Json -Compress)" -ForegroundColor Red
            $failed++
        }
    } catch {
        $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
        $detail = $_.Exception.Message
        # Intentar leer el body del error para mostrar más info de Grafana.
        if ($_.Exception.Response) {
            try {
                $stream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object IO.StreamReader($stream)
                $body = $reader.ReadToEnd()
                if ($body) { $detail = $body }
            } catch {}
        }
        Write-Host " FALLA (HTTP $code): $detail" -ForegroundColor Red
        $failed++
    }
}

# --- Paso 3: resumen ---
Write-Host ""
Write-Host "[3/3] Resumen: $ok OK, $failed con error" -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
Write-Host ""
if ($ok -gt 0) {
    Write-Host "Dashboards disponibles en:" -ForegroundColor Cyan
    Write-Host "  $GrafanaUrl/dashboards"
}
if ($failed -gt 0) {
    exit 1
}
