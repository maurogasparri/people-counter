# Plan oficial del proyecto — 12 sprints

**Proyecto:** Sistema de conteo de personas para retail (Levi's / Leuru Group)
**Stack:** Raspberry Pi 5 + Hailo-8L + IMX708 estéreo + WiFi/BLE + AWS
**Modelo de trabajo:** Claude Code escribe el código bajo dirección humana; el humano dirige, decide arquitectura y valida con hardware
**Duración:** 12 sprints semanales
**Esfuerzo total estimado:** 118.5 horas

---

## Resumen de los 12 sprints

| Sprint | Épica | Foco | Horas |
|:---:|:---:|---|---:|
| S1 | EP-00 | Análisis y diseño inicial | 7 |
| S2 | EP-01 | Captura estéreo y servicios | 8 |
| S3 | EP-02 | Calibración estéreo | 15 |
| S4 | EP-03 | Profundidad y región de interés | 6 |
| S5 | EP-04 | Detección neuronal de personas | 14 |
| S6 | EP-05 | Seguimiento y conteo | 9 |
| S7 | EP-06 | Captura WiFi y BLE | 11 |
| S8 | EP-07 | Mensajería y telemetría | 7 |
| S9 | EP-08 | Servicios cloud y APIs | 13 |
| S10 | EP-09 | Visualización analítica | 9 |
| S11 | EP-10 | Validación y documentación | 11.5 |
| S12 | EP-11 | Cierre del prototipo | 8 |
| | | **Total** | **118.5** |

---

## S1 — EP-00 Análisis y diseño inicial (7h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Especificaciones funcionales (RF-01 a RF-12) y no funcionales (RNF-01 a RNF-14) | 2 | Documento de especificaciones |
| Diseño general de arquitectura HW + SW + cloud | 2 | Diagramas de arquitectura |
| BOM y adquisición de componentes | 1 | Lista de materiales |
| Setup del repositorio con CI básica | 1 | `pyproject.toml`, configuración ruff/pytest, `.gitignore`, README inicial |
| Estructura inicial del proyecto | 1 | `src/`, `tests/`, `docs/`, `scripts/`, `infra/`, `config/` con `__init__.py` |

## S2 — EP-01 Captura estéreo y servicios (8h)

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

## S3 — EP-02 Calibración estéreo (15h)

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

## S4 — EP-03 Profundidad y región de interés (6h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Pipeline SGBM con post-filtro WLS | 2 | `src/vision/depth.py` + tests |
| Wizard diagnose_depth | 1 | `scripts/diagnose_depth.py` |
| Wizard diagnose_bracket (verificación mecánica) | 1 | `scripts/diagnose_bracket.py` |
| Validación con escena real de laboratorio | 2 | Mapas de profundidad calibrados |

## S5 — EP-04 Detección neuronal de personas (14h)

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

## S6 — EP-05 Seguimiento y conteo (9h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Filtro de Kalman para tracking | 1 | `src/tracking/kalman.py` + tests |
| Tracker con asociación estilo ByteTrack | 2 | `src/tracking/tracker.py` + tests |
| Line crossing counter con ROI | 1.5 | `src/tracking/counter.py` + tests |
| ROI picker interactivo (define line y zona) | 1 | `scripts/roi_picker.py` |
| World coords + clasificación adulto/niño | 0.5 | `src/vision/world_coords.py` |
| Static suppressor (filtro de FP sobre clutter) | 1 | `src/vision/static_suppressor.py` |
| Generador de reportes de eventos | 0.5 | `src/vision/report.py` |
| Pruebas integrales del pipeline visual | 1.5 | Conteo correcto sobre baseline frames |

## S7 — EP-06 Captura WiFi y BLE (11h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Análisis del problema MAC randomization + evaluación de técnicas publicadas | 2 | Decisión documentada de las 3 reglas |
| Captura WiFi en modo monitor con nexmon | 1.5 | `src/wifi_ble/wifi_probe.py` |
| Captura BLE pasiva con bleak | 1 | `src/wifi_ble/ble_scan.py` |
| Anonymizer con hash SHA-256 + sal diaria | 0.5 | `src/wifi_ble/hasher.py` + tests |
| Stitching con las 3 reglas sobre hash_groups | 2.5 | `src/wifi_ble/dedup.py` + tests |
| Publisher de resúmenes cada 15 min | 1 | `src/wifi_ble/publisher.py` |
| Service systemd dedicado | 0.5 | `config/wifi-monitor.service` |
| Exportador anonimizado para auditoría | 0.5 | `scripts/export_anonymized.py` |
| Validación en hardware con tráfico real | 1.5 | Stitching ratio dentro de rango esperado |

## S8 — EP-07 Mensajería y telemetría (7h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Cliente MQTT con TLS mutuo y reconexión + backoff | 1.5 | `src/mqtt/client.py` + tests |
| Buffer SQLite local con retención 72h | 1.5 | `src/mqtt/buffer.py` + tests |
| Device Shadow base con whitelist | 1 | Integración con cliente MQTT |
| Telemetría operativa cada 5 min | 1 | `src/status/monitor.py` |
| Integración con pipeline (orquestación en main) | 1 | `main.py` con telemetría |
| Tests E2E del flujo MQTT con simulación de desconexión | 1 | Reentrega validada |

## S9 — EP-08 Servicios cloud y APIs (13h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Diseño detallado del stack cloud | 1.5 | Decisión arquitectural |
| CloudFormation phaseado (VPC, security groups, NAT) | 2 | `infra/cloudformation/people-counter.yaml` (sección networking) |
| RDS PostgreSQL Multi-AZ con snapshots + PITR | 1 | Sección RDS del CFN |
| ECS Fargate + ALB + ACM (HTTPS con custom domain) | 1.5 | Sección compute del CFN |
| AWS IoT Core + Topic Rules + certificados X.509 | 1 | Sección IoT del CFN |
| Lambda persist_event + IAM mínimo | 1 | `src/cloud/persist_event.py` + tests |
| Esquema SQL events + vistas materializadas | 1 | `infra/sql/bootstrap.sql` |
| Acceso programático a datos del sistema (cierra US-08) | 1 | Usuario `readonly_external` con SELECT sobre vistas; `docs/api-access.md` con ejemplos de queries; whitelist de IPs en SG |
| **API REST de ingest de datos POS (cierra US-06)** | **2** | **Schema `pos_transactions`, Lambda `ingest_pos_transaction`, API Gateway con IAM auth, vista `conversion_rate_by_store`** |
| Deploy scripts (sh + ps1) | 0.5 | `scripts/deploy_lambda.sh`, `.ps1`, `infra/deploy.ps1` |
| Política documentada de backups y DR | 0.5 | Sección en `docs/` |

## S10 — EP-09 Visualización analítica (9h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Grafana sobre Fargate (provisioning del servicio) | 1 | Incorporado en CFN |
| Diseño visual de los 4 dashboards | 1.5 | Wireframes y decisiones de UX |
| Vista 1 — Dashboard consolidado de la red | 1 | JSON del dashboard provisioning |
| Vista 2 — Detalle de sucursal con drill-down | 1 | JSON del dashboard provisioning |
| Vista 3 — Monitoreo de la flota | 1 | JSON del dashboard provisioning |
| Vista 4 — Reportes exportables (CSV export nativo) | 1 | JSON del dashboard provisioning |
| Alerting configurado para US-02 (umbrales por sucursal) | 1 | Reglas de Grafana alerting |
| CloudWatch monitoring + dashboards básicos | 0.5 | Métricas y alarmas |
| Integration tests de Lambda + bucket de eventos | 1 | `tests/cloud/test_persist_event.py` ampliado |

## S11 — EP-10 Validación y documentación (11.5h)

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
| **Documentación del procedimiento de migración de datos históricos** | **0.5** | **`docs/historical-data-migration.md`, `scripts/templates/migrate_historical.py`** |

## S12 — EP-11 Cierre del prototipo (8h)

| Tarea | Horas | Artefactos |
|---|---:|---|
| Hardening final del repositorio (ruff sweeps, type hints) | 1.5 | Repo limpio |
| Coverage final de tests | 0.5 | Reporte de coverage |
| Provisioning script para nuevas unidades | 1 | `scripts/provision.py` |
| Dimensionamiento del despliegue (para sección 6.2 TFG) | 2 | Tabla de dimensionamiento |
| Demo en video | 1.5 | Screencast del sistema completo |
| Capturas de pantalla para el TFG | 0.5 | Imágenes para anexo |
| Entregables finales del TFG | 1 | Documento final entregado |

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

## Mapeo épica → US habilitadas

| Épica | US habilitadas | US que cierra |
|---|---|---|
| EP-00 Análisis | --- | --- |
| EP-01 HW + captura | US-09 (datos) | --- |
| EP-02 Calibración | US-01, US-05 (precondición) | --- |
| EP-03 Profundidad | US-05 (precondición) | --- |
| EP-04 Detección | US-05 (datos) | --- |
| EP-05 Tracking | US-01, US-05 (datos) | --- |
| EP-06 WiFi/BLE | US-04 (datos) | --- |
| EP-07 Mensajería | US-09 (datos), US-10, US-11 | **US-10, US-11** |
| EP-08 Cloud + APIs | US-01, US-03, US-06, US-07, US-09 (precondición), US-08 | **US-06** (T9.11 ingest POS), **US-08** (T9.8 acceso readonly PostgreSQL) |
| EP-09 Visualización | --- | **US-01, US-02, US-03, US-04, US-05, US-07, US-09** |
| EP-10 PoC + docs | (validación cruzada de todas) | --- |
| EP-11 Cierre | --- | --- |

---

## Instrucciones para Claude Code (reorganización del repo)

Usar este plan como referencia oficial para revisar y reordenar el historial del repositorio.

**Tareas sugeridas para Claude Code:**

1. **Mapear cada commit a su tarea del plan**: por cada commit del repo, identificar a qué tarea de qué sprint pertenece su contenido principal.

2. **Identificar tandas para squash**: agrupar commits consecutivos sobre la misma tarea (especialmente UX iteration en wizards) en un único commit por tarea.

3. **Reescribir mensajes de commit ambiguos**: para los commits que tengan mensajes vagos (ej: "Update README and CLAUDE.md"), reescribir con un mensaje claro que refleje la tarea del plan correspondiente.

4. **Reordenar cronológicamente por sprint**: el orden ideal del historial es S1 → S2 → S3 → ... → S12, con los commits agrupados dentro de cada sprint por tarea.

5. **Para commits que toquen múltiples tareas o sprints**: el criterio por defecto es **dividir** el commit en múltiples commits separados con `git rebase -i` + `edit` + `git reset HEAD~1` + commits parciales, uno por tarea/sprint, para que el esfuerzo quede correctamente atribuido en cada lugar. La excepción es cuando la división complica la legibilidad sin agregar valor: en esos casos se puede mantener el commit unificado, pero el mensaje debe explicitar **claramente qué partes pertenecen a cada tarea/sprint** y el esfuerzo correspondiente a cada uno se imputa al sprint dominante (no se duplica).

6. **Unificar commits de actividad principal con sus commits de documentación rezagada**: durante el desarrollo es frecuente hacer un *commit* de código y olvidar actualizar la documentación asociada, generando un *commit* posterior de "docs: actualizar X" o "README: reflejar Y". Estos *commits* de documentación rezagada deben fusionarse (squash) con el *commit* de la actividad principal que los origina, dado que conceptualmente forman parte del mismo trabajo. Aplica especialmente a:
   - *Commits* de actualización de `README.md` / `CLAUDE.md` que siguen a un *commit* de feature
   - *Commits* de actualización de guías en `docs/` que siguen a cambios en código relacionado
   - *Commits* de actualización de `config.example.yaml` que siguen a cambios en `src/config/`
   - *Commits* de actualización de tests que siguen al *commit* del módulo testeado (si fueron olvidados originalmente)

   El *commit* resultante debe quedar bajo el *scope* de la actividad principal, no bajo `docs:`.

7. **Validar que cada tarea del plan tenga al menos un commit asociado**: si alguna tarea queda sin commits, marcarlo como gap (probablemente sea actividad humana no codificable como labeling, captura de dataset, iteración con tablero físico, decisiones de arquitectura, etc.).

**Format de commit recomendado (Conventional Commits con scope):**

```
<scope>: <descripción concisa de la tarea del plan>

[contexto opcional sobre qué cambió y por qué]
[referencia al sprint y tarea del plan si conviene]
```

Ejemplos:
- `vision: implementar pipeline ChArUco fisheye (S3, EP-02)`
- `tracking: tracker con asociación ByteTrack-style (S6, EP-05)`
- `wifi_ble: stitching con 3 reglas sobre hash_groups (S7, EP-06)`
- `infra: CloudFormation phaseado con Fargate + ALB (S9, EP-08)`
