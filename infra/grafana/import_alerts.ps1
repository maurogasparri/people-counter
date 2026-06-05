<#
.SYNOPSIS
    Importa las reglas de alerta de infra/grafana/alerting/alert-rules.json a
    Grafana via la Provisioning API. Idempotente: sobrescribe por rule-group.

.DESCRIPTION
    Flow:
      1. Obtiene el UID de la datasource ``people-counter`` (Postgres).
      2. Asegura la carpeta de alertas (uid fijo ``alertas-pc``, título "Alertas").
      3. Sustituye el placeholder ``__DS_UID__`` por el UID real.
      4. Por cada grupo: PUT /api/v1/provisioning/folder/<uid>/rule-groups/<grupo>
         con el header ``X-Disable-Provenance`` (así las reglas quedan editables
         desde la UI, no bloqueadas como "provisionadas").

    Auth: igual que import_dashboards.ps1 (GRAFANA_API_TOKEN o basic admin).

.EXAMPLE
    .\infra\grafana\import_alerts.ps1
#>

param(
    [string]$GrafanaUrl = "https://grafana.tfg.gasparri.com.ar",
    [string]$AlertsFile = ""
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $AlertsFile) {
    $AlertsFile = Join-Path $SCRIPT_DIR "alerting\alert-rules.json"
}
if (-not (Test-Path $AlertsFile)) {
    throw "No existe el archivo de alertas: $AlertsFile"
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
Write-Host "  Import de alert rules Grafana"  -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "URL:     $GrafanaUrl"
Write-Host "Archivo: $AlertsFile"
Write-Host "Auth:    $authMode"
Write-Host ""

# --- Paso 1: UID de la datasource ---
Write-Host "[1/3] Obteniendo UID de la datasource people-counter..." -ForegroundColor Cyan
$ds = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/name/people-counter" -Headers $authHeader -TimeoutSec 15
$dsUid = $ds.uid
Write-Host "  UID: $dsUid" -ForegroundColor Green

# --- Paso 2: carpeta de alertas (uid fijo, título ASCII) ---
$spec = Get-Content $AlertsFile -Raw -Encoding UTF8 | ConvertFrom-Json
$folderUid = $spec.folderUid
$folderTitle = $spec.folderTitle

Write-Host ""
Write-Host "[2/3] Asegurando carpeta '$folderTitle' (uid=$folderUid)..." -ForegroundColor Cyan
$folders = Invoke-RestMethod -Uri "$GrafanaUrl/api/folders" -Headers $authHeader -TimeoutSec 15
$existing = $folders | Where-Object { $_.uid -eq $folderUid }
if ($existing) {
    # Renombrar si el título difiere (ej. "Alertas — People Counter" -> "Alertas").
    if ($existing.title -ne $folderTitle) {
        $body = @{ title = $folderTitle; overwrite = $true } | ConvertTo-Json -Compress
        $tmp = Join-Path $env:TEMP "grafana-alertfolder.json"
        [IO.File]::WriteAllBytes($tmp, [Text.Encoding]::UTF8.GetBytes($body))
        Invoke-RestMethod -Uri "$GrafanaUrl/api/folders/$folderUid" -Method Put -Headers $authHeader `
            -ContentType "application/json; charset=utf-8" -InFile $tmp -TimeoutSec 15 | Out-Null
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        Write-Host "  Renombrada -> '$folderTitle'" -ForegroundColor Green
    } else {
        Write-Host "  Ya existe." -ForegroundColor Green
    }
} else {
    $body = @{ uid = $folderUid; title = $folderTitle } | ConvertTo-Json -Compress
    $tmp = Join-Path $env:TEMP "grafana-alertfolder.json"
    [IO.File]::WriteAllBytes($tmp, [Text.Encoding]::UTF8.GetBytes($body))
    Invoke-RestMethod -Uri "$GrafanaUrl/api/folders" -Method Post -Headers $authHeader `
        -ContentType "application/json; charset=utf-8" -InFile $tmp -TimeoutSec 15 | Out-Null
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    Write-Host "  Creada." -ForegroundColor Green
}

# --- Paso 3: push de cada rule-group ---
Write-Host ""
Write-Host "[3/3] Importando rule-groups..." -ForegroundColor Cyan
$alertHeader = $authHeader.Clone()
$alertHeader["X-Disable-Provenance"] = "true"  # reglas editables en la UI

$ok = 0; $failed = 0
foreach ($grp in $spec.groups) {
    Write-Host "  - $($grp.name) ($($grp.rules.Count) reglas)..." -NoNewline
    try {
        $payload = @{ interval = $grp.interval; rules = $grp.rules }
        $json = ($payload | ConvertTo-Json -Depth 100) -replace "__DS_UID__", $dsUid
        $tmp = Join-Path $env:TEMP "grafana-rulegroup.json"
        [IO.File]::WriteAllBytes($tmp, [Text.Encoding]::UTF8.GetBytes($json))
        $grpPath = [uri]::EscapeDataString($grp.name)  # nombres con acento (ej. Operación)
        Invoke-RestMethod -Uri "$GrafanaUrl/api/v1/provisioning/folder/$folderUid/rule-groups/$grpPath" `
            -Method Put -Headers $alertHeader -ContentType "application/json; charset=utf-8" `
            -InFile $tmp -TimeoutSec 30 | Out-Null
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        Write-Host " OK" -ForegroundColor Green
        $ok++
    } catch {
        $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
        $detail = $_.Exception.Message
        if ($_.Exception.Response) {
            try {
                $reader = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
                $b = $reader.ReadToEnd(); if ($b) { $detail = $b }
            } catch {}
        }
        Write-Host " FALLA (HTTP $code): $detail" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "Resumen: $ok OK, $failed con error" -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
if ($failed -gt 0) { exit 1 }
