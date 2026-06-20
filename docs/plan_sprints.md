# Plan oficial del proyecto — 12 sprints

**Proyecto:** Sistema de conteo de personas para retail (Levi's / Leuru Group)
**Stack:** Raspberry Pi 5 + Hailo-8L + IMX708 estéreo + WiFi/BLE + AWS
**Modelo de trabajo:** Claude Code escribe el código bajo dirección humana; el humano dirige, decide arquitectura y valida con hardware
**Duración:** 12 sprints semanales
**Esfuerzo total estimado:** 119.5 horas
**Extensión post-PoC** (fuera de este plan): rollout de flota (OTA + HA cloud) — ver el anexo al final.

---

## Resumen de los 12 sprints

| Sprint | Foco | Horas |
|:---:|---|---:|
| S1 | Análisis y diseño inicial | 7 |
| S2 | Captura estéreo y servicios | 8 |
| S3 | Calibración estéreo | 15 |
| S4 | Profundidad y región de interés | 6 |
| S5 | Detección neuronal de personas | 14 |
| S6 | Seguimiento y conteo | 10 |
| S7 | Captura WiFi y BLE | 11 |
| S8 | Mensajería y telemetría | 7 |
| S9 | Servicios cloud y APIs | 13 |
| S10 | Visualización analítica | 9 |
| S11 | Validación y documentación | 11.5 |
| S12 | Cierre del prototipo | 8 |
| | | **Total** | **119.5** |

---

## S1 — Análisis y diseño inicial (7h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Especificaciones funcionales (RF-01 a RF-13) y no funcionales (RNF-01 a RNF-12) | 2 | Documento de especificaciones |
| Diseño general de arquitectura HW + SW + cloud | 2 | Diagramas de arquitectura |
| BOM y adquisición de componentes | 1 | Lista de materiales |
| Setup del repositorio con CI básica | 1 | `pyproject.toml`, configuración ruff/pytest, `.gitignore`, README inicial |
| Estructura inicial del proyecto | 1 | `src/`, `tests/`, `docs/`, `scripts/`, `infra/`, `config/` con `__init__.py` |

## S2 — Captura estéreo y servicios (8h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Ensamblaje físico del dispositivo | 1 | RPi5 + AI HAT+ + IMX708 ×2 + PoE HAT + gabinete |
| Bootstrap RPi OS Trixie 64-bit | 1 | SO funcional con SSH habilitado |
| Captura estéreo dual con picamera2 raw mode | 1.5 | `src/vision/capture.py` + tests |
| Sistema de configuración con back-compat | 1 | `src/config/loader.py`, `src/config/hardware.py`, `scripts/migrate_config.py` |
| Status LED + health monitor | 1 | `src/status/led.py`, `src/status/health.py`, `scripts/test_led.py` |
| Services systemd + logrotate + purge | 1 | `config/people-counter.service`, `config/logrotate.conf`, `scripts/purge_best_frames.py` y timer asociado |
| Setup script del dispositivo | 0.5 | `scripts/setup_device.sh` |
| Validación end-to-end de la captura sobre HW | 1 | Frames raw guardados correctamente |

## S3 — Calibración estéreo (15h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Pipeline ChArUco con modelo Kannala-Brandt para fisheye | 2 | `src/vision/calibration.py` + tests |
| Caracterización del sensor (modos canónicos IMX708) | 1.5 | Documentación interna + valores en `config/hardware.yaml` |
| Wizard focus_assist (asistente de enfoque óptico) | 2 | `scripts/focus_assist.py` con UI navegable |
| Wizard calibrate (asistente guiado de calibración) | 3 | `scripts/calibrate.py` con TTS, poses, ground truth |
| Wizard preview (preview rectificada) | 1 | `scripts/preview.py` |
| Captura de baseline frames para bench | 1 | `scripts/capture_baseline_frames.py` |
| Rescale analítico de calibraciones | 0.5 | `scripts/rescale_calibration.py` |
| Viewer navegable con gating | 1 | `src/web/viewer.py` |
| Iteración con tablero físico hasta convergencia | 3 | Calibración válida con reproj < 1 px |

## S4 — Profundidad y región de interés (6h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Pipeline SGBM con post-filtro WLS | 2 | `src/vision/depth.py` + tests |
| Wizard diagnose_depth | 1 | `scripts/diagnose_depth.py` |
| Monitor de salud de calibración (post-cal, error epipolar) | 1 | `scripts/diagnose_calibration.py` |
| Validación con escena real de laboratorio | 2 | Mapas de profundidad calibrados |

## S5 — Detección neuronal de personas (14h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Diseño del approach (head detection vs full body, resolución, etc.) | 1 | Decisión arquitectural documentada |
| Captura multi-site del dataset desde NVRs externas | 1 | `scripts/training/capture_mjpeg.py` |
| Labeling asistido en Roboflow (Smart Polygon) | 2 | Dataset etiquetado |
| Labeling helper con smoothing del live preview | 1 | `src/web/annotate.py` |
| Toolkit de training (record, sample, convert) | 1.5 | `scripts/training/record_clips.py`, `sample_for_roboflow.py`, `polys_to_bboxes.py`, `_embed_calib_into_sites.py` |
| Notebook de entrenamiento YOLOv8n en Kaggle T4 | 1 | `scripts/training/train_head_detector.ipynb` |
| Entrenamiento efectivo en Kaggle (supervisión activa) | 1.5 | Modelo entrenado |
| Evaluación del modelo | 1 | `eval_yolo.py`, `analyze_eval_summary.py` |
| Compilación HEF en toolkit Hailo | 1 | Modelo HEF deployable |
| Integración runtime del detector | 2 | `src/vision/detect.py`, `src/vision/best_frame.py` |
| Bench del detector en hardware | 1 | `scripts/training/bench_detector.py`, throughput validado |

## S6 — Seguimiento y conteo (10h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Filtro de Kalman para tracking | 1 | `src/tracking/kalman.py` + tests |
| Tracker con asociación 2-stage + ghost pool + decay tunable + Lowe ratio | 2 | `src/tracking/tracker.py` + tests. Incluye state machine CANDIDATE→CONFIRMED→PENDING→LOST, two-stage matching (high+low conf), ghost pool con ID adoption (IoU+dist gates), `pending_velocity_decay`, `last_observed_position` separado del Kalman pushed |
| Line crossing counter con counting zone + net-balance + death-emit con guards | 1.5 | `src/tracking/counter.py` + tests. Net-balance de cruces por visita counting zone, gate de cruce solo con detección real, decisive Kalman cross al exit, death-emit-if-crossed con guards (`had_outside_pos` + `MIN_VISIT_RANGE_FOR_DEATH_EMIT`), debounce |
| counting zone picker interactivo (define line y zona) | 1 | `scripts/counting_zone_picker.py` |
| World coords + clasificación adulto/niño | 0.5 | `src/vision/world_coords.py` |
| Static suppressor + exempt counting zone (filtro de FP sobre clutter fuera de la counting zone) | 1 | `src/vision/static_suppressor.py`. Exime la counting zone de conteo (los tracks dentro no se filtran) para no perder lingerers reales |
| Generador de reportes de eventos | 0.5 | `src/vision/report.py` |
| Pruebas integrales del pipeline visual | 1.5 | Conteo correcto sobre baseline frames |
| Live preview HTTP/MJPEG para validación operativa | 1 | `src/web/viewer.py` + `src/web/annotate.py` + tests. Composite L\|R\|disparidad servido vía MJPEG; overlay de tracks (bbox fijo cuadrado, trayectoria 30-frames, id, height), reconexión robusta al restart del pipeline, panel de stats (counts, hourly, exterior). Usado por el operador durante el piloto |

## S7 — Captura WiFi y BLE (11h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Análisis del problema MAC randomization + evaluación de técnicas publicadas | 2 | Decisión documentada de las 3 reglas |
| Captura WiFi en modo monitor con nexmon | 1.5 | `src/wifi_ble/wifi_probe.py` |
| Captura BLE pasiva con bleak | 1 | `src/wifi_ble/ble_scan.py` |
| Anonymizer con hash SHA-256 + sal diaria | 0.5 | `src/wifi_ble/hasher.py` + tests |
| Stitching con las 4 reglas sobre hash_groups | 2.5 | `src/wifi_ble/dedup.py` + `src/wifi_ble/fingerprint.py` + tests. Reglas: (1) seqnum continuity 802.11 (anti MAC-rotation), (2) cross-protocol L2 (WiFi+BLE simultáneo), (3) BLE anchoring (lifetime de RPA ~15min iOS), (4) fingerprint continuity (orden de IEs + manufacturer-data, sobrevive rotación de Apple H1+ que resetea seqnum). MAX-RSSI para clasificación passersby/shoppers. Filtro `randomized_only` (solo dispositivos humanos) |
| Publisher de resúmenes cada 15 min | 1 | `src/wifi_ble/publisher.py` |
| Service systemd dedicado | 0.5 | `config/wifi-monitor.service` |
| Exportador anonimizado para auditoría | 0.5 | `scripts/export_anonymized.py` |
| Validación en hardware con tráfico real | 1.5 | Stitching ratio dentro de rango esperado |

## S8 — Mensajería y telemetría (7h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Cliente MQTT con TLS mutuo y reconexión + backoff | 1.5 | `src/mqtt/client.py` + tests |
| Buffer SQLite local con retención 72h | 1.5 | `src/mqtt/buffer.py` + tests |
| Device Shadow base con whitelist | 1 | Integración con cliente MQTT |
| Telemetría operativa cada 5 min + canaries | 1 | `src/status/monitor.py` + telemetry payload en `src/main.py`. OS health (cpu/hailo temp, disk, mem), pipeline (fps, latencies, dets/tracks), MQTT (connected, disconnect_count, buffer_backlog), WiFi/BLE health + `wifi_ble_stitching_ratio`, **canaries del tracker**: `track_stitching_ratio` (unique IDs / counts emitidos, ideal ≈1.0; >1.3 = fragmentación) y `death_emit_count` (fallback firings; diferencia "fragmenta-y-rescata" de "fragmenta-y-pierde"). Persistido vía Lambda en columnas RDS dedicadas |
| Integración con pipeline (orquestación en main) | 1 | `main.py` con telemetría |
| Tests E2E del flujo MQTT con simulación de desconexión | 1 | Reentrega validada |

## S9 — Servicios cloud y APIs (13h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Diseño detallado del stack cloud | 1.5 | Decisión arquitectural |
| CloudFormation phaseado (VPC, security groups, NAT) | 2 | `infra/cloudformation/people-counter.yaml` (sección networking) |
| RDS PostgreSQL Multi-AZ con snapshots + PITR | 1 | Sección RDS del CFN |
| ECS Fargate + ALB + ACM (HTTPS con custom domain) | 1.5 | Sección compute del CFN |
| AWS IoT Core + Topic Rules + certificados X.509 | 1 | Sección IoT del CFN |
| Lambda persist_event + IAM mínimo | 1 | `src/cloud/persist_event.py` + tests |
| Esquema SQL events + vistas materializadas | 1 | `infra/sql/bootstrap.sql` |
| Acceso programático a datos del sistema vía SQL directo (cierra US-08, RF-12) | 1 | Usuario `readonly_external` con SELECT sobre vistas; `docs/api_access.md` con ejemplos de queries; whitelist de IPs en SG |
| **API REST de ingest de datos POS (cierra US-06, RF-11)** | **2** | **Schema `pos_transactions`, Lambda `ingest_pos_transaction`, API Gateway con IAM auth, vista `conversion_rate_by_store`** |
| **API REST de consulta de agregados (cierra US-09, RF-13)** | **2** | **Lambda `query_aggregates.py` con cursor pagination opaco + RFC 8288 Link header + RFC 7807 errors + ETag/Cache-Control + OpenAPI 3.1 servido en `/v1/openapi.json` + EMF metrics; rol `lambda_query_reader` (IAM auth, SELECT-only); `docs/api_access.md` con `curl --aws-sigv4` examples** |
| Deploy scripts (sh + ps1) | 0.5 | `scripts/deploy_lambda.sh`, `.ps1`, `infra/deploy.ps1` |
| Política documentada de backups y DR | 0.5 | Sección en `docs/` |

## S10 — Visualización analítica (9h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Grafana sobre Fargate (provisioning del servicio) | 1 | Incorporado en CFN |
| Diseño visual de los 4 dashboards | 1.5 | Wireframes y decisiones de UX |
| **Vista 1 — Operaciones (PRIORIDAD)**: KPIs por sucursal (footfall, in/out, turn-in rate, conversion) | 1 | JSON del dashboard provisioning |
| Vista 2 — Detalle de sucursal con drill-down | 1 | JSON del dashboard provisioning |
| Vista 3 — Monitoreo de la flota (canaries del tracker + uptime devices) | 1 | JSON del dashboard provisioning. **Hecho**: tablero ⑤ "Salud de la flota" (carpeta Operación y flota) con los 3 canaries (`track_stitching_ratio`, `ghost_adoption_count`, `death_emit_count`) + estado de devices, frescura, temperatura, FPS, backlog MQTT y errores. |
| Vista 4 — Reportes exportables (CSV export nativo) | 1 | JSON del dashboard provisioning |
| Alerting configurado para US-02 (umbrales por sucursal) | 1 | Reglas de Grafana alerting versionadas en `infra/grafana/alerting/alert-rules.json` + `import_alerts.ps1`. **Hecho**: 12 reglas en 3 grupos en la carpeta "Alertas", con contact point email. **Hardware** (8): CPU > 80 °C, throttle/undervolt, fan clavado, FS read-only, crash-loop, reloj sin NTP, cámara caída, WiFi mudo. **Operación** (3): sucursal sin datos por frescura, `track_stitching_ratio > 1.3`, `death_emit_count / total > 0.3` (con piso de ≥10 counts/día anti-ruido). **Negocio** (1): pico de tráfico por franja (US-02). Único pendiente: definir con el cliente el canal/destinatario final de notificación (decisión de negocio, no de código — el contact point email ya está cableado a `mauro@gasparri.com.ar`). |
| CloudWatch monitoring + dashboards básicos | 0.5 | Métricas y alarmas |
| Integration tests de Lambda + bucket de eventos | 1 | `tests/cloud/test_persist_event.py` ampliado |

## S11 — Validación y documentación (11.5h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Banco de pruebas en laboratorio (setup físico) | 1.5 | Escenarios definidos |
| Ejecución de los 15 casos de prueba (TC-01 a TC-15) | 2.5 | Reporte de resultados |
| Integración E2E con todos los módulos | 1 | `tests/test_integration_e2e.py` |
| Smoke tests del stack completo | 0.5 | `tests/test_main.py` |
| Preflight checks | 1 | `scripts/preflight.py` |
| Verificación de hardware | 0.5 | `scripts/verify_hardware.py` |
| Replay mode del pipeline (testing offline) | 0.5 | `main.py --replay-dir` |
| Guía de setup del dispositivo | 0.75 | `docs/setup_guide.md` |
| Guía de calibración en laboratorio | 0.75 | `docs/lab_calibration_guide.md` |
| Guía operativa para piloto | 1 | `docs/pilot_operator_guide.md` |
| Declaración de privacidad | 0.5 | `docs/privacy.md` |
| Project gantt interno | 0.5 | `docs/project_gantt.md` |
| **Procedimiento de migración de datos históricos** | **0.5** | **`scripts/migrate_historical.py` (loader CSV→staging batcheado) + `infra/sql/migrate_historical_rollups.example.sql` (transform staging→tablas base rollup_*). La doc del procedimiento vive como comentario-cabecera en ambos archivos.** |
| **Hardening anti-FP durante piloto (sesión 2026-05-24)** | **6** | **9 commits sobre `src/tracking/counter.py`, `src/tracking/tracker.py`, `src/vision/pre_filter.py` (NEW), `src/web/annotate.py`, `src/web/viewer.py`, config, runbook, matrix. Incluye: rename `counter.roi → counting_zone`; expone 5 knobs del rescue cascade config-driven; nuevos guards `min_count_height_m` / `min_real_inside_frames` / `height_confidence_gate`; filtro pre-tracker `tracking_zone` con modos `polygon` / `frame_margin_px` / `auto_margin_px`; keepalive condicional a entry real (opción E); fix doble-conteo `last_outside_pos` stale; blur del preview fuera de tracking_zone; `/health` endpoint + auto-reload del MJPEG. 922 tests verde.** |
| **Tooling de setup para espacios chicos / luz difícil (2026-06-19, ~6.5h)** | **6.5** | **`scripts/diagnose_calibration.py` (NEW, reemplaza `diagnose_bracket` — salud post-cal por error epipolar); `calibrate.py` modo barrido (sweep) default + `focus_assist.py` modo mapa default, ambos con reporte HTML. Tests: diagnose_calibration (12) + sweep (13) + foco-mapa (6) + calibrate --manual (3). Validado en HW: barrido → baseline 143.2mm + 0.59px epipolar out-of-sample; mapa de foco PASS. Docs sincronizadas (README/CLAUDE/lab_guide/setup_guide).** |

## S12 — Cierre del prototipo (8h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Hardening final del repositorio (ruff sweeps, type hints) | 1.5 | Repo limpio |
| Coverage final de tests | 0.5 | **Hecho**: `docs/coverage_report.md` — 1039 tests passed (+2 skipped), **82% de cobertura total** (núcleo counter/tracker/dedup/calibración 88–98%; bordes de hardware 60–77% cubiertos por validación on-device). Pasada de hardening 2026-06-17: guards del death-emit aislados (counter 97%) + resiliencia de las Lambdas (persist_event / ingest_pos 98%) + fix de una race latente en el test del viewer. Regenerable con `pytest --cov=src`. |
| Provisioning script para nuevas unidades | 1 | `scripts/provision.py` |
| Dimensionamiento del despliegue (para sección 6.2 TFG) | 2 | Tabla de dimensionamiento |
| Demo en video | 1.5 | Screencast del sistema completo |
| Capturas de pantalla para el TFG | 0.5 | Imágenes para anexo |
| Entregables finales del TFG | 1 | Documento final entregado |

## Anexo — Extensión post-PoC: Rollout de flota (OTA + HA cloud) (~58h)

**Fuera del plan original de 12 sprints.** Iniciativa post-PoC: lo que vuelve el
sistema operable a escala sin visitas a sitio. Diseño completo de OTA en
[`docs/ota_design.md`](ota_design.md). OTA = AWS IoT Jobs + S3 (presigned URL) +
firma Ed25519 + swap atómico A/B con rollback automático + canario observado en ⑤ +
ventana de horario cerrado.

| Tarea | Horas | Artefactos |
|---|---:|---|
| Diseño OTA (decisiones de arquitectura) | ✅ | `docs/ota_design.md` |
| Infra AWS: S3 bucket + IoT Jobs + thing-groups + firma | 8–12 | CFN ampliado |
| Build/CI: empaquetar + firmar + subir + crear Job | 4–6 | `scripts/ota_publish.*` |
| Agente de Jobs en el device (paho): lifecycle + download + verify | 12–18 | `src/ota/agent.py` |
| Swap atómico A/B + ventana + preservación de estado | 8–12 | layout `/opt/people-counter/`, `setup_device.sh` |
| Health-check + rollback automático | 6–10 | integrado con health monitor |
| Versión en telemetría (`app_version`/`model_version`/`last_ota_status`) + panel skew en ⑤ | 2–3 | schema telemetry + tablero |
| Testing: harness de fallas + E2E en HW (canario, update malo) | 10–18 | tests + validación en Pi |
| HA cloud (backlog): RDS Multi-AZ + Route53 delegated + Managed Grafana | — | ver `CLAUDE.md` |

---

## Asunciones del modelo de estimación

- **Claude Code escribe el código** (incluyendo tests unitarios) bajo dirección del humano
- **El humano dirige, revisa, decide arquitectura y valida con hardware**
- Los bucles que dependen de hardware/datos físicos NO se aceleran con Claude Code:
  - Calibración con tablero ChArUco físico
  - Captura de dataset de imágenes para training
  - Tiempos de entrenamiento en Kaggle (con supervisión activa)
  - Validación end-to-end del PoC
- El **labeling de dataset** y la **escritura de documentación técnica** tampoco se aceleran
- Las decisiones de arquitectura (Fargate vs ECS, RDS vs Aurora, las 3 reglas de stitching) requieren tiempo humano de análisis

## Mapeo sprint → US habilitadas

| Sprint | US habilitadas | US que cierra |
|---|---|---|
| S1 — Análisis | --- | --- |
| S2 — HW + captura | US-10 (datos) | --- |
| S3 — Calibración | US-01, US-05 (precondición) | --- |
| S4 — Profundidad | US-05 (precondición) | --- |
| S5 — Detección | US-05 (datos) | --- |
| S6 — Tracking | US-01, US-05 (datos) | --- |
| S7 — WiFi/BLE | US-04 (datos) | --- |
| S8 — Mensajería | US-10 (datos), US-11, US-12 | **US-11, US-12** |
| S9 — Cloud + APIs | US-01, US-03, US-06, US-07, US-10 (precondición), US-08, US-09 | **US-06** (T9.11 ingest POS), **US-08** (T9.8 acceso SQL readonly), **US-09** (T9.12 API REST de consulta) |
| S10 — Visualización | --- | **US-01, US-02, US-03, US-04, US-05, US-07, US-10** |
| S11 — PoC + docs | (validación cruzada de todas) | --- |
| S12 — Cierre | --- | --- |

---

## Convención de commits (aplicada)

El historial del repo fue reorganizado el **2026-05-23** para alinearlo con este plan: 508 commits originales → **122 commits** vía squash por `(fecha, sprint primario)`, con prefijo de sprint/tarea obligatorio en el subject. El tree del HEAD es bit-identical al pre-rewrite (zero pérdida de código). El detalle de la reorganización vive en [docs/commit_mapping.md](commit_mapping.md).

**Formato canónico del subject de commit:**

```
[S<N>/T<X>] <type>(<scope>): <descripción concisa>

[body opcional con contexto]

Co-Authored-By: ...
```

| Caso | Ejemplo |
|---|---|
| Single task | `[S6/T6.3] fix(counter): debounce de jitter en line crossing` |
| Multi-task mismo sprint | `[S6/T6.2,T6.3] feat(tracking): ghost pool + ID adoption` |
| Multi-sprint | `[S6/T6.3][S8/T8.5] feat(tracking): canary track_stitching_ratio` |
| Docs subordinados al feature | `[docs] docs(plan): regenerar mapping post-rewrite` |

**Reglas operativas hacia adelante:**

1. **Prefijo `[S<N>/T<X>]` obligatorio** en todo commit nuevo. Sprints ordenados ASC, tasks ASC dentro de cada bloque, sin "winner" — el commit es un container de contribuciones.
2. **Pocos commits por día**. Idealmente 1 por sprint tocado. Iteraciones intra-día se acumulan con `git commit --amend` antes del primer push, no spamean el log.
3. **Docs subordinados** (`README.md`, `CLAUDE.md`, guías de un feature ya commiteado) usan prefijo `[docs]` y se commitean junto al feature parent o como follow-up explícito.
4. **Cero menciones de productos externos en el repo** (memorias internas separadas vivían en `memory/` gitignored y siguen ahí). El sanitizer del rewrite las removió de los mensajes históricos.

El backup del estado pre-rewrite quedó como tag local `pre-rewrite-20260523-143821` (no pusheado al remote para mantener el repo público prolijo). El tree de los 508 commits originales sigue accesible vía ese tag por si en el futuro hace falta consultar SHAs viejos.
