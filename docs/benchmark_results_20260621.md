# Resultados de benchmark — Grupo A (banco) — 2026-06-21

> **Estado:** Grupo A (mediciones de hardware en banco, sin montaje cenital).
> El **Grupo B** (cruces reales en geometría cenital) queda listado al final como
> pendiente. Reporte para el Trabajo Final de Grado.

## Encabezado

| Campo | Valor |
|---|---|
| Fecha/hora | 2026-06-21, ~13:00–16:00 (-03) |
| Commit del repo | `8e08bc5` (`git rev-parse --short HEAD` en la Pi y en el repo) |
| Hardware edge | Raspberry Pi 5 + AI HAT+ (Hailo-8L) + 2× IMX708 (estéreo, baseline 140 mm) + Waveshare PoE HAT (G) 25,5 W + Active Cooler |
| RAM física del banco | **8 GB** (unidad de desarrollo). El spec de flota es **2 GB**; el headroom se evalúa contra 2 GB, no contra los 8 GB del banco (ver `docs/hardware_sizing.md`, validación por ballooning) |
| microSD del banco | ~116 GB libre (la unidad de banco no usa la SanDisk 64 GB del spec) |
| SO / Python | RPi OS Trixie 64-bit / Python 3.13.5 |
| Pipeline | `python3 -m src.main` (servicio `people-counter.service`, `User=pi`) |
| Calibración cargada | `/etc/people-counter/calibration.npz` (11-may-2026) |
| Cloud | RDS Postgres 16.13 (`db.t4g.micro`), IoT Core, Lambda `persist_event`, ECS Fargate/Grafana |
| Condición general | Banco (no cenital). Escena mayormente vacía; detecciones incidentales de una persona estática durante la ventana A5 |

**Nota de método para potencia:** `scripts/measure_power.py` suma V×I de **todos** los
rieles del PMIC (output-side; incluye placa + Hailo vía PCIe + cámaras). El draw
de pared se estima +13 % por pérdidas de conversión de la fuente PoE. El Hailo
**no** expone potencia propia (no hay `hailortcli measure-power`): su consumo
está incluido dentro del total del PMIC, **no es medible por separado**.

---

## A1 — Consumo eléctrico

Método: `scripts/measure_power.py --duration N --interval 0.5 --rails --csv …`
(0,5 s/muestra). CSV crudos: `power_session.csv` (running idle-throttled),
`power_fullthroughput.csv` (running full), `power_idle.csv` (servicio detenido).

| Estado | n muestras | avg (W) | p50 (W) | p95 (W) | máx (W) | mín (W) | pared +13% avg | % PoE 25,5 W (avg) |
|---|---|---|---|---|---|---|---|---|
| (a) pipeline **detenido** (idle real) | 572 | **2,16** | 2,15 | 2,19 | 3,06 | 2,11 | 2,44 | 8,5 % |
| (b) corriendo, escena vacía, **idle-throttle ON** (~10–16 FPS) | 8483 | **4,00** | 3,79 | 5,38 | 9,93 | 2,82 | 4,52 | 15,7 % |
| (b′) corriendo, escena vacía, **full-throughput** (~28 FPS) | 571 | **4,67** | 4,38 | 6,57 | 8,75 | 3,36 | 5,28 | 18,3 % |

Rieles dominantes (estado b): `VDD_CORE` 1,21 W · `3V3_SYS` 1,18 W · resto < 0,4 W c/u.
Cruce con telemetría del device (`power_w`, 48 h): avg 4,06 W, rango 3,44–4,9 W —
**consistente** con `measure_power.py`. (El pico aislado de 5,99 W observado en una
única muestra inicial era un transitorio; el promedio sostenido es ~4 W.)

**Hallazgo:** el consumo es casi independiente de la carga de cómputo — pasar de
10 FPS (idle-throttle) a 28 FPS (full) sube el promedio sólo ~0,67 W; el pipeline
agrega ~1,8 W (throttled) a ~2,5 W (full) sobre el baseline del SO (2,16 W). El
pico absoluto observado fue 9,93 W (transitorio), **< 40 % del presupuesto de
25,5 W**.

**Proyección energética** (estado b, 12 h/día × 363 días): 4,00 W → **17,4 kWh/año**
output-side; ~**19,7 kWh/año** de pared.

---

## A2 — Térmico

Método: carga sostenida (pipeline full + `stress-ng --cpu 4` + loopers `/health`
+ readers MJPEG) ≥ 10 min. CSV: `stress_monitor.csv`. Temp Hailo vía telemetría
(`hailo_temp_c`, probe `hailortcli` del health monitor).

| Métrica | Valor observado | Límite | Veredicto |
|---|---|---|---|
| Temp CPU máx bajo estrés (con cooler) | **64,8 °C** (start 57,1 → avg 62,9) | 80 °C | ✅ (15 °C de margen) |
| Temp Hailo (telemetría, actual / 48 h máx) | 39,5 °C / **41,8 °C** | 85 °C | ✅ |
| Throttling (`get_throttled`) bajo estrés (132 muestras) | **0x0 en todas** | 0x0 | ✅ nunca throttleó |
| Throttling en 48 h de operación (telemetría) | `throttled_flags=0` siempre | 0 | ✅ |

Carga del test: pipeline full (28 FPS) + `stress-ng --cpu 4` (4 cores al 97,8 %
promedio) + 4 loopers `/health` + 3 readers MJPEG, durante 660 s. Aún con la CPU
saturada el cooler mantuvo 64,8 °C máx (muy por debajo de 80 °C) y **nunca hubo
throttling**.

Sin Active Cooler: **no re-ejecutado** (requiere desconexión física del fan +
riesgo de throttle). Caracterización previa en `docs/thermal_characterization.md`
(equilibrio fan-OFF 83–84 °C, throttle pero FPS aguanta por ser Hailo-bound).

---

## A3 — Memoria

Método: soak de operación sostenida 60,4 min (sampler 5 s, `soak_system.csv`) +
stress sintético de laboratorio. Memoria del servicio vía cgroup v2
(`memory.current`/`.peak`/`.events`).

| Métrica | Valor observado | Umbral | Veredicto |
|---|---|---|---|
| `memory.current` servicio (soak) | 403–411 MiB (avg 408) | — | — |
| `memory.peak` servicio (soak) | **416 MiB** | < 2048 (target flota) | ✅ (20 % usado, 1632 MiB margen) |
| `memory.events` (high/max/oom/oom_kill) | **todos 0** | 0 | ✅ |
| Swap usado (vmstat/meminfo) | **0** (sin swapping) | 0 | ✅ |
| `MemAvailable` sistema (mín) | 7420 MiB | — | (unidad de 8 GB) |
| `memory.peak` bajo stress sintético (proceso fresco) | **247 MiB** | < 2048 | ✅ (12 % del target) |
| `memory.events` bajo stress (high/max/oom) | **todos 0** | 0 | ✅ |

**Nota honesta:** `memory.current` (cgroup) cuenta también page-cache atribuido al
cgroup, por eso 408–416 MiB > los ~270 MB de *RSS working set* documentados en
`hardware_sizing.md` (que mide RSS anónimo). Ambas cifras son correctas para lo
que miden; el working set anónimo entra holgado en 2 GB. El stress es **sintético
de laboratorio** (CPU saturada + N conexiones concurrentes), no tráfico real de
tienda.

---

## A4 — CPU

Método: `/proc/stat` (deltas), `/proc/loadavg`. (`mpstat` no instalado.)

| Estado | CPU (% agregado 4 cores) | Por-core | loadavg(1m) | FPS |
|---|---|---|---|---|
| Nominal, idle-throttle ON (soak) | avg 15,3 % · p95 27,4 % · máx 43,8 % | — | avg 0,94 · máx 2,66 | ~10–16 |
| Full-throughput, escena vacía | ~24 %/core (**~96 % en términos de `top`, ≈ 1 core**) | cpu0-3: 22–26 % | ~1,0 | ~28 |
| Bajo stress sintético (4 cores saturados) | avg 97,8 % · **máx 99,8 %** | todos ~100 % | máx 12,7 | cae a **14–20** |

**¿Qué satura primero?** En operación nominal **ninguno**: el pipeline es
serial/single-core (~1 de 4 cores), con 3 cores ociosos y RAM lejísimos del
límite. El cuello de botella del FPS es la **latencia serial por frame** (captura
+ inferencia + SGBM), no la CPU agregada ni la RAM. Bajo stress sintético la CPU
sí se satura (ver A4 stress). Relación FPS↔CPU: ~28 FPS ≈ 1 core.

---

## A5 — FPS y latencia por etapa  *(lo más importante)*

Método: instrumentación existente `--profile --profile-every-n 1` (gated, default
off — Regla 1). Ventana 300 s, 8346 frames. Parser: `parse_profile.py`. CSV por
frame: `profile_empty_perframe.csv`. La etapa **publicación** se cubre en A6
(publish MQTT asíncrono, costo per-frame ~0 salvo en eventos de conteo).
Precisión per-stage: ±1 ms (el profiler loguea ms enteros).

**FPS efectivo sostenido = 27,8 FPS** (ventana mayormente vacía).

Escena **vacía** (`det=N`, SGBM se omite — sólo corre con detecciones), 8118 frames:

| Etapa | media (ms) | p50 | p95 | % frame |
|---|---|---|---|---|
| captura estéreo | 14,1 | 14 | 17 | 40 % |
| rectificación | 4,9 | 5 | 6 | 14 % |
| inferencia Hailo | 15,9 | 16 | 17 | 46 % |
| profundidad (SGBM) | 0,0 | 0 | 0 | 0 % (omitida) |
| tracking | 0,0 | 0 | 0 | 0 % |
| **TOTAL** | **34,9** | 35 | 37 | → 28,8 FPS |

**Con detección** (`det=Y`, SGBM corre), 228 frames:

| Etapa | media (ms) | p50 | p95 | % frame |
|---|---|---|---|---|
| captura estéreo | 5,1 | 0 | 17 | 10 % |
| rectificación | 4,3 | 4 | 5 | 9 % |
| inferencia Hailo | 15,2 | 15 | 17 | 31 % |
| **profundidad (SGBM)** | **22,6** | 26 | 30 | **46 %** |
| tracking | 2,2 | 2 | 4 | 5 % |
| **TOTAL** | **49,3** | 48 | 66 | → 21,4 FPS |

Latencia por frame (telemetría del device): `frame_latency_p50_ms` = **24,0 ms**,
`frame_latency_p95_ms` = **26,5 ms**.

**Hallazgos:** (1) el SGBM es la etapa más pesada cuando hay gente (~46 % del
frame, p95 30 ms); con escena vacía se omite por completo → FPS sube de ~21 a ~28.
(2) La etapa de captura a full-throughput es mayormente *espera de la cámara*
(~14 ms ≈ período del sensor); por eso cae a ~5 ms cuando el SGBM ocupa el frame
(el siguiente par ya está buffereado). El pipeline es **cámara-bound (~28 FPS)
con escena vacía y SGBM-bound (~21 FPS) con gente**.

---

## A6 — Latencia extremo a extremo (device → RDS)

Método: `query_e2e_live.py` (boto3 lee secret `rds-master`, psycopg a RDS).
`received_at - event_ts` (s), **device real** `store-pilot-01-cam-01`, últimas 48 h,
ventana estable (lag 0–120 s; el replay de cortes y los datos demo/seed se excluyen).
CSV: `e2e_latency_live.csv`.

| Tabla | n | p50 (s) | p95 (s) | mín (s) | máx (s) | lag negativo (reloj) |
|---|---|---|---|---|---|---|
| `count_events` | 2 | 0,246 | 0,253 | 0,237 | 0,254 | 0 |
| `telemetry` | 20 | 0,252 | 13,68 | 0,208 | 82,93 | 0 |
| `wifi_ble_events` (`period_end`) | 82 | 0,297 | 0,586 | 0,231 | 0,586 | 0 |

**Sincronización de reloj:** `timedatectl` → System clock synchronized = yes, NTP
active. **0 lags negativos** en el device real → el reloj del device nunca va
adelantado del server (sano). El p95 alto de telemetry (13,7 s, máx 82,9 s) es 1
outlier de 20 (probable reconexión momentánea / cold-start de Lambda); el p50 es
sub-segundo en las tres tablas.

---

## A7 — Calibración estéreo

Método: `diagnose_calibration.py` (error epipolar, no necesita ground-truth).

| Métrica | Valor | Umbral | Veredicto |
|---|---|---|---|
| RMS epipolar (px) | **INCONCLUSO** — 0/8 frames con board (no había tablero en cuadro en el banco) | ≤ 1 px sano | ⏳ requiere tablero |
| Baseline ‖T‖ de la calib cargada | **146,6 mm** (diseño 140 mm → +6,6 mm de drift calibrado) | ±1–2 mm del diseño | ⚠️ drift conocido |
| fx (P1) de la calib cargada | 460,7 px @ 1152×648 | — | — |

`diagnose_calibration.py` corrió y cargó la calibración (11-may-2026) pero no
detectó el tablero ChArUco (0/8 frames) → el error epipolar no se pudo medir.
Salud previa conocida (`project_calib_deferred_lighting`): ~2 px epipolar
in-service / 0,59 px out-of-sample en el barrido; recalibración **ya diferida**
por flicker del lab. El baseline calibrado (146,6 mm) está +6,6 mm sobre el diseño
de 140 mm — drift consistente con la recalibración pendiente. El MAE de estatura
en geometría cenital real → **Grupo B (B3)**.

---

## A8 — Arranque en frío y sincronización de cámaras

| Métrica | Valor | Objetivo | Veredicto |
|---|---|---|---|
| Cold-start: boot kernel+userspace (`systemd-analyze`) | 2,1 s + 26,8 s = **28,9 s** | — | — |
| Cold-start: reboot → frames fluyendo (`Pipeline: FPS`) | **+38 s** | < 90 s | ✅ |
| Cold-start: reboot → MQTT conectado | +30 s | — | ✅ |
| Servicios activos tras boot | people-counter / wifi-monitor / reset.timer = **active** | todos up | ✅ |
| Sync estéreo (% pares L/R ≤ 5 ms, 8392 pares / 300 s) | **87,79 %** (p50 0,37 ms · p95 18,8 ms · máx 18,9 ms · mean 2,6 ms) | alto | ⚠️ ver nota |

**Nota sync (hallazgo honesto):** las dos IMX708 se capturan en paralelo pero, en la
config actual, **no hay sync de frame activo**. El 88 % de los pares cae < 5 ms
(p50 0,37 ms), pero ~12 % llega hasta ~un período de frame (~19 ms) por la deriva
de fase entre los relojes de ambos sensores. A 3 m/s, 19 ms ≈ 6 cm — el cap
`max_exposure_us=16000` (16 ms) acota el blur. Distribución bimodal: la mayoría
sub-ms, una cola a ~18,8 ms. CSV: `camsync.csv`.

**PoC de sync por software (probado el 2026-06-21):** libcamera/picamera2 (0.7.0 /
0.3.34) exponen `controls.rpi.SyncModeEnum` (Off/Server/Client) — sync de frames
**sin cableado** entre las dos cámaras del mismo Pi (una Server emite timing, la
otra Client ajusta el largo de frame). Medido en este device (1152×648 @ 30 FPS,
right=Server / left=Client): el delta L/R cae de **8,4 ms (sin sync)** a
**0,025 ms p50 / 0,060 ms máx (25–60 µs), 100 % de pares < 1 ms**. Elimina la cola.
Script: `sync_poc.py`.

**Integración — 3 condiciones (todas validadas en HW, `sync_poc.py`/`_diag*.py`):**
La integración reveló que el sync necesita las 3, no alcanza con una:
1. **Config de video** (`create_video_configuration`): con `create_still_configuration`
   el IPA de sync NO ajusta el frame timing → queda estancado en ~18 ms.
2. **`FrameRate` fijo y ALCANZABLE** (NO un rango en `FrameDurationLimits`): el
   rango tira `Sync algorithm enabled with variable framerate` (error por frame),
   y un rate mayor al que el sensor da (1e6/16000=62,5 fps > piso ~56 fps) NO
   converge (queda ~9 ms). Solución: leer el piso real del sensor
   (`FrameDurationLimits[0]`≈17,8 ms) tras el start y setear `FrameRate=1e6/
   (piso·1,05)`≈53 fps (exposure ~18,1 ms, +0,3 ms vs piso). Medido: FR56→71 µs,
   FR53→43 µs, FR50→18 µs.
3. **converge-then-hold** (lo más sutil): el IPA hace su ajuste inicial de fase
   SÓLO si el dequeue tiene GAPS — la captura continua del productor lo IMPIDE
   (queda ~18 ms desde el frame 1, validado con 7 diagnósticos incl.
   `capture_sync_request`). Pero una vez convergido con gaps, el productor
   continuo lo MANTIENE (holdea a ~17-41 µs). Solución: una fase de convergencia
   bloqueante (~8-15 s, bursts con pausas) ANTES de arrancar el productor.

Receta final: `SyncMode` (left=Server/principal, right=Client, client primero) +
`create_video_configuration` + `apply_sync_framerate()` (FrameRate del piso del
sensor) + fase `_converge_sync()` / `converge_camera_sync()`, detrás del flag
`vision.camera_sync.enabled` (default off). **Feature de RUNTIME** (+ calibrate
sweep + preview live, donde hay movimiento; focus/diagnose/baseline NO lo usan —
escena estática). **INTEGRADO** en `src/vision/capture.py` + activado en el piloto
(`camera_sync_converged delta_us=…` por boot; cold-start ~+10s). Scripts del PoC y
los 8 diagnósticos en `docs/benchmarks/20260621/sync_*.py`.

---

## A9 — Detector en el dispositivo

Método: `hailortcli run` del HEF en el Hailo-8L (throughput puro on-chip).

| Métrica | Valor | Veredicto |
|---|---|---|
| Throughput (inf/s) | **105,9 FPS** (533 frames, `hailortcli run`) | ✅ ~5× el rate del pipeline |
| mAP/precisión/recall on-device | **no medido** — `hailortcli run` usa datos random; no hay set de validación + harness de eval en la Pi | — |

El Hailo-8L corre el `people-counter-detector.hef` a **105,9 inf/s** puro on-chip,
muy por encima de los ~21–28 FPS que consume el pipeline → el NPU **no es el cuello
de botella** (la etapa `detect` del pipeline mide ~16 ms = 62 FPS incluyendo
pre/post-proceso + overhead de VStream; el chip puro va a 106). La calidad INT8
(mAP50 0,96 del detector v2 desplegado) se validó en el pipeline de entrenamiento
contra el val set held-out (ver `scripts/training/README.md`), no on-device.

---

## A10 — Resiliencia

| Caso | Resultado | Veredicto |
|---|---|---|
| TC-11/12 corte de conectividad (buffer local + drena íntegro) | bloqueo egress MQTT (nft, tcp:8883) 5,5 min → telemetría buffereada (`unsent` 0→1, `telemetry_published mqtt=False`); al restaurar: `MQTT conectado` + `Buffer replay: 1 messages sent` → `unsent` 0 (~50 s). Sin pérdida; dup en replay cubierto por la UNIQUE (TC-15) | ✅ PASS |
| TC-13 reanudación tras reboot (proxy de corte) | reboot → todos los servicios up + frames fluyendo en **+38 s** | < 90 s | ✅ PASS |

**Nota TC-13:** se midió con un reboot limpio (recovery 38 s). Un corte de energía
real (yank de PoE) agregaría la renegociación PoE (~segundos) + posible fsck en
apagado sucio, pero el camino de recuperación de systemd es idéntico. Yank físico
real → opcional, queda disponible para corroborar.

---

## A11 — Crecimiento de almacenamiento

Método: snapshot de tamaños al inicio/fin del soak (~1 h 15 min).

| Archivo | Inicio | Fin | Δ | Nota |
|---|---|---|---|---|
| `buffer.db` (outbox MQTT) | 40 960 B | 81 920 B | +40 KB | Crece por páginas SQLite; drena tras PUBACK → **acotado** |
| `wifi_ble_dedup.sqlite` | 929 792 B | 929 792 B | 0 | Se resetea diario (rotación de salt) → estado estacionario ~1 MB |
| `/var/lib/people-counter` total | 5,16 MB | 5,21 MB | +40 KB | — |

**Proyección microSD:** el crecimiento persistente es mínimo y acotado (outbox
drena, dedup resetea diario, journald está size-capped). Sin riesgo de llenado de
la microSD en operación sostenida. La acumulación histórica vive en RDS (cloud),
no en la tarjeta.

---

## A12 — Funcionales que no requieren cenital

| Caso | Criterio | Resultado | Veredicto |
|---|---|---|---|
| TC-14 Privacidad | sin imágenes/video ni PII persistida | best_frame **deshabilitado**; 0 archivos jpg/png/mp4/raw en disco; **0 MACs crudas** en journal; dedup guarda sólo hashes salteados (16 B) + `group_id` UUID opaco + fingerprint hasheado + RSSI + salt rotado | ✅ PASS |
| TC-15 Idempotencia cloud | re-ingesta no duplica | re-INSERT de un `count_event` existente → **0 filas** (ON CONFLICT), count 1160→1160 | ✅ PASS |
| TC-08/09 WiFi/BLE stitching | dispositivos de prueba conocidos | observado (sin dispositivos controlados): `wifi_ble_stitching_ratio`=0,15 (stitching activo y agresivo); 471 hashes / 238 agrupados | ⚠️ Parcial (falta corrida con devices conocidos) |
| TC-17 POS | ingesta + conversión | API desplegada (`https://api.tfg.gasparri.com.ar/pos/transactions`) | ⏳ no probada esta corrida |
| TC-18 API de consulta | params válidos/inválidos + auth | Lambda `query-aggregates` existe pero **no expuesta por HTTP** (sólo el API de POS tiene API Gateway) | ⚠️ N/A vía HTTP |

---

## Tabla resumen de casos de prueba

| Caso | Criterio de aceptación | Resultado observado | Veredicto |
|---|---|---|---|
| TC-08/09 stitching | merge de identidad | ratio 0,15; sin devices controlados | ⚠️ Parcial |
| TC-11/12 buffer offline | drena íntegro | buffer 0→1 en corte, replay drena a 0 en ~50 s, sin pérdida | ✅ PASS |
| TC-13 corte energía | reanuda < 90 s | reboot → operativo en 38 s (yank físico real pendiente opcional) | ✅ PASS (proxy) |
| TC-14 privacidad | sin imágenes/PII | 0 imágenes, 0 MACs crudas, sólo hashes salteados | ✅ PASS |
| TC-15 idempotencia | sin duplicados | 0 filas en re-insert | ✅ PASS |
| TC-17 POS | conversión | API existe, no probada | ⏳ |
| TC-18 API consulta | params + auth | Lambda existe, sin HTTP | ⚠️ N/A |
| **TC-01** ingreso | ≥ 95 % | — | ⏳ **Grupo B** |
| **TC-02** egreso | ≥ 95 % | — | ⏳ **Grupo B** |
| **TC-03** multidirección | ≥ 90 % | — | ⏳ **Grupo B** |
| **TC-04** filtro altura | 100 % | `min_count_height_m=1.0` verificado en config | ⏳ **Grupo B** (cruce real) |
| **TC-05** estatura ±10 cm | ≥ 90 % | — | ⏳ **Grupo B** |
| **TC-06** par cancelado | ≥ 90 % | — | ⏳ **Grupo B** |
| **TC-07** sin omisión/doble | ≥ 95 % | — | ⏳ **Grupo B** |

---

## Pendiente — Grupo B (requiere montaje cenital + cruces reales)

- **B1 — Exactitud de conteo controlado (TC-01…TC-07):** entrada conocida vs tasa
  de acierto, en geometría cenital sobre el punto de paso.
- **B2 — Consumo "con gente cruzando":** tercer estado de A1; validar contra
  `counting_event_published`; confirmar independencia de carga.
- **B3 — MAE de estatura en geometría cenital real:** validación fiel de A7 con
  cámaras en posición/altura definitivas.

---

## Anexo — Método (comandos exactos)

| Medición | Comando |
|---|---|
| Commit | `git rev-parse --short HEAD` |
| A1 potencia | `python3 scripts/measure_power.py --duration N --interval 0.5 --rails --csv <out>` |
| A2 térmico | `stress-ng --cpu 4 --timeout 660s` + sampler `vcgencmd measure_temp/get_throttled/measure_clock arm` |
| A3 memoria | soak: `sampler.sh <out> 720 5`; cgroup `cat /sys/fs/cgroup/system.slice/people-counter.service/memory.{current,peak,events}` |
| A4 CPU | deltas de `/proc/stat` (agregado y per-core) + `/proc/loadavg` |
| A5 perfil | drop-in `ExecStart … --profile --profile-every-n 1` + `journalctl … \| grep PROFILE` → `parse_profile.py` |
| A6 e2e | `py docs/benchmarks/20260621/query_e2e_live.py` (boto3 + psycopg, RDS) |
| A7 calibración | `python3 scripts/diagnose_calibration.py` |
| A8 cold-start/sync | reboot + timestamp del primer frame; `SensorTimestamp` L/R |
| A9 detector | `hailortcli run models/people-counter-detector.hef` |
| A10 resiliencia | bloqueo de egress MQTT (iptables) + watch `buffer.db`; reboot |
| A11 storage | `stat -c %s` de los SQLite al inicio/fin |
| A12 privacidad/idempotencia | `inspect_dedup.py`, `query_telemetry.py` (TC-15) |

CSV crudos y scripts reproducibles en `docs/benchmarks/20260621/`.

---

# Actualización — Re-test de sync estéreo + automatización de casos de prueba

> Segunda corrida (mismo día, commit del feature de sync `221eb21`+). Banco, sin
> montaje cenital. Re-mide el sync L/R tras la implementación y cierra por
> automatización los TC que no dependen de cruces físicos.

## Parte 1 — Re-test de sincronización estéreo (antes/después)

Método: `camsync_v2.py` (StereoCapture `camera_sync=True` + `async_capture=True`,
igual que el runtime: converge-then-hold, luego captura continua), 17 618 pares /
330 s. CSV: `camsync_v2.csv`. Base: `camsync.csv` (sin sync).

| Métrica | Base (sin sync) | Nueva (con sync) | Criterio | Veredicto |
|---|---|---|---|---|
| mediana delta_ms | 0,37 | **0,025** | — | — |
| p95 delta_ms | 18,8 | **0,050** | ≤ 5 ms | ✅ |
| máx delta_ms | 18,9 | **4,19** | — | ✅ |
| std delta_ms | — | **0,261** | — | — |
| % pares ≤ 5 ms | 87,8 % | **100,00 %** | > 99 % | ✅ |
| modo episódico ~19 ms | ~12 % | **0 %** | desaparece | ✅ |

Histograma nuevo: 93,97 % en [0–0,05 ms), 3,26 % [0,05–0,1), cola despreciable,
**0 pares ≥ 15 ms** (el modo a media trama desapareció). **El sync cumple el
criterio**: p95 = 50 µs (≤ 5 ms) y 100 % de los pares ≤ 5 ms.

## Parte 2 — Casos de prueba automatizados

| Caso | Criterio | Resultado observado | Veredicto |
|---|---|---|---|
| Calibración epipolar | RMS ≤ 1 px | **RMS epipolar 2,48 px** (12/12 frames, 396 corners, máx 2,77 px, disparidad mediana 68 px ≈ 0,99 m). Calib 11-may ‖T‖=146,6 mm → la geometría drifteó | ❌ No cumple → **recalibrar** |
| TC-08 stitching WiFi | MACs de 1 device → 1 grupo | 6 MACs randomizadas (seqnum continuo + fingerprint) → **1 group_id**; otro device → grupo aparte (`tc08_09_stitching.py`) | ✅ Cumple |
| TC-09 stitching WiFi+BLE | mismo device 2 protocolos → 1 grupo | WiFi+BLE <2s ΔRSSI 2 dBm → **mismo grupo**; control ΔRSSI 38 dBm → grupo aparte | ✅ Cumple |
| TC-11 corte breve | retransmisión íntegra | bloqueo nft 8883 → telemetría buffereada → replay drena (0 pérdida/0 dup) — corrida en vivo §A10 | ✅ Cumple |
| TC-12 corte prolongado | 72 h sin pérdida/dup, sin desborde | 1 302 eventos (72 h) offline persistidos → drenados 0 pérdida / 0 dup; cap acota a 1 000 (`tc12_buffer_72h.py`) | ✅ Cumple |
| TC-14 privacidad | 0 imágenes, 0 PII en claro | disco: 0 jpg/png/mp4; journal: **0 MACs crudas**; RDS: `visitor_hash` opaco (16 B bytea), 0 con formato MAC; sin columnas de imagen/PII (`query_tc.py`) | ✅ Cumple |
| TC-15 idempotencia (count) | sin duplicados | re-INSERT de count_event existente → 0 filas (§A12) | ✅ Cumple |
| TC-17 POS idempotencia | persistencia única + conversión | re-INSERT de pos_transaction existente → 0 filas (7122→7122); ventas/día computables para conversion_rate (`query_tc.py`) | ✅ Cumple |
| TC-18 API de consulta | válida ok / inválida y sin-auth rechazadas | invoke Lambda `query-aggregates`: válida (6d) → **200** con agregados; falta `from` → **400** missing-parameter; rango >7d → **400** range-too-large (RFC-7807). Sin-auth = N/A (no expuesta por HTTP; auth iría en API GW) | ✅ Cumple (auth N/A) |
| TC-13 reanudación tras corte de energía | < 90 s, buffer íntegro, fs sano | **corte físico real**: boot 3,1s+33,8s; servicio +36s, MQTT +37s, **frames +46 s (< 90 s)**; `integrity_check` buffer.db + dedup.sqlite = **ok**; fs root `rw`, 0 errores dmesg, NRestarts 0, throttle 0x0 (sin undervolt); sync re-convergió (972 µs); telemetría resumió en RDS (4 eventos nuevos) | ✅ Cumple |

Respaldo: las 4 suites unitarias de los componentes (`test_dedup`, `test_buffer`,
`test_ingest_pos_transaction`, `test_query_aggregates`) → **115 passed**.

**TC-01…TC-07** (conteo controlado): **pendientes — requieren montaje cenital**.

**Nota sobre la calibración (epipolar 2,48 px):** la calib guardada (11-may)
drifteó por encima del umbral de salud (1 px) — consistente con la recalibración
ya diferida por flicker del lab. **Recomendación: recalibrar (barrido + foco)
ANTES del MAE de profundidad del Grupo B (B3)**, para que el MAE refleje una
calibración sana y no el drift actual.

### Anexo de método (Parte 2)

| Test | Comando |
|---|---|
| Sync v2 | `python3 camsync_v2.py` (en la Pi, servicio detenido) |
| TC-08/09 | `py docs/benchmarks/20260621/tc08_09_stitching.py` |
| TC-12 | `py docs/benchmarks/20260621/tc12_buffer_72h.py` |
| TC-14/17 | `py docs/benchmarks/20260621/query_tc.py` (RDS) |
| TC-18 | `aws lambda invoke --function-name people-counter-query-aggregates-dev --payload fileb://q_{valid,invalid}.json` |
| Suites | `pytest tests/wifi_ble/test_dedup.py tests/mqtt/test_buffer.py tests/cloud/test_{ingest_pos_transaction,query_aggregates}.py` |
| Epipolar | `PYTHONPATH=. python3 scripts/diagnose_calibration.py --frames 12` (con tablero ChArUco a ~1 m) |
| TC-13 corte físico | pre: snapshot buffer/RDS; corte real del riel PoE; post: `uptime -s` + `systemd-analyze` + offsets de journal desde boot + `PRAGMA integrity_check` + `findmnt / ` + `vcgencmd get_throttled` + telemetría en RDS |

---

# Cierres automatizables — costo cloud, runner, seguridad, cobertura, TC

> Tercera corrida (mismo día, commit `1762ac7`+). AWS en **solo lectura**. Cinco
> items que no requieren montaje cenital. Crudos en `docs/benchmarks/20260621/`
> (+ `raw/` del runner). Entorno: Windows 11 · Python 3.12.6.

## 1. Costo cloud (Cost Explorer, solo lectura)

`cloud_cost.py` (boto3 `ce:GetCostAndUsage`). Se reporta el **uso BRUTO**
(`RECORD_TYPE=Usage`, costo de lista) porque el **neto facturado es ~$0**:
free-tier + créditos cubren el 100% del uso (junio: Usage +44,9 / Credit −44,9).

**Escenario PROTOTIPO actual** (instancia única, sin HA), uso bruto por servicio
(junio 2026, primer mes con todo el stack arriba; extrapolado a 30 d):

| Servicio | Bruto (USD/mes) | Nota |
|---|---|---|
| ECS Fargate (Grafana) | ~20,4 | vs $18 estimado |
| Elastic Load Balancing (ALB) | ~15,7 | ≈ $16 estimado |
| VPC (IPv4 pública) | ~13,9 | **NO figuraba en la estimación TFG** |
| RDS Postgres (db.t4g.micro) | ~13,4 | ≈ $13 estimado |
| Secrets Manager | ~0,8 | |
| IoT / Lambda / ECR | < 0,1 | free-tier |
| **TOTAL bruto** | **~64** | mayo (mes de ramp-up parcial): ~28 |
| **NETO facturado** | **~0** | free-tier + créditos |

| Columna | Escenario | USD/mes |
|---|---|---|
| Medido (neto) | prototipo hoy | **~0** (free-tier/créditos) |
| Medido (bruto/lista) | prototipo hoy | **~64** |
| Estimación TFG | prototipo | ~35–49 |
| Proyección producción (HA) | RDS Multi-AZ (+$13) | **~77** fijo |

- **Diferencia vs TFG**: el bruto real (~$64) supera la estimación (~$35–49),
  principalmente por **IPv4 pública del VPC (~$14/mo)** que la estimación omitió
  (AWS cobra direcciones IPv4 públicas desde 2024). Anual prototipo (bruto):
  ~$768. Producción HA: ~$924/año.
- **Flota de 38 (cloud COMPARTIDO)**: RDS/ALB/ECS/VPC son fijos para toda la
  flota; sólo IoT/Lambda/storage escalan (marginal, <$1/device/mes). Estimación
  flota ≈ ~$77 fijo + ~$38 marginal ≈ **~$115/mes para 38 = ~$3/device/mes
  amortizado** (vs ~$64/device con 1 solo). El costo por dispositivo CAE fuerte
  con el tamaño de flota. Crudos: `aws_cost_*.json`. **→ modelo preciso en §1b.**

### 1b. Modelo de costo de lista preciso (Price List API, base 730 h)

`pricing_full.py` (fijos) + `pricing_variables.py` (variables). Tarifas `[API]`
desde el Price List API; `[lista]` = tarifa publicada us-east-1 para servicios con
muchos SKUs por tier que el filtro del API no desambigua (Fargate, IoT, API GW,
CloudWatch). `estimate-template-cost` se descartó (legacy, retirado 2023, rechaza
`AWS::SES::EmailIdentity`). El **neto facturado sigue ~$0** (free-tier + créditos);
esto es el costo de LISTA.

**Fijos** (1 deployment; cloud compartido por la flota):

| Línea | Prototipo (micro SAZ) | Producción HA (small MAZ) |
|---|---|---|
| Fargate (proto 0,5/1 ×1 · prod **1vCPU/2GB ×2**) `[lista]` | $18,02 | $72,08 |
| ALB + 1 LCU `[API]` | $22,27 | $22,27 |
| RDS instancia `[API]` | $11,68 | $47,45 |
| RDS storage gp3 `[API]` | $2,30 | $11,50 (50GB ×2 MAZ) |
| Secrets Manager ×2 `[API]` | $0,80 | $0,80 |
| IPv4 pública ×4 `[API]` | $14,60 | $14,60 |
| **Fijo/mes** | **$69,67** | **$168,70** |

**Variables** (flota 38, Tabla 12) — recalculado vs Tabla 17 (§6.4.2):

| Componente | Recalc | Tabla 17 | Δ |
|---|---|---|---|
| IoT Core (2M msgs + rules) `[lista]` | $2,30 | $3,00 | −0,70 |
| Lambda (2M inv, 256MB, ~100ms) `[API]` | $1,15 (lista) | $0,50 | +0,65 |
| API Gateway (~1M req, supuesto) `[lista]` | $1,00 | $3,00 | −2,00 |
| CloudWatch (12 alarmas + ~1GB logs, supuesto) `[lista]` | $1,70 | $5,00 | −3,30 |
| Transferencia egress (360 MB) `[API]` | **$0,00** (100GB free) | $5,00 | −5,00 |
| SNS (alertas) `[lista]` | $0,00 (1000 free) | $1,00 | −1,00 |
| **Subtotal variable** | **$6,15** | $17,50 | **−11,35** |

→ **Tabla 17 sobreestima ~$11/mes** (sobre todo egress $5→$0: 360 MB entra en el
free-tier de 100 GB; CloudWatch $5→$1,7; SNS $1→$0). IoT/Lambda casi clavados.

**Totales (lista):**

| Escenario | Fijo | Variable | **Total/mes** | Anual | $/device |
|---|---|---|---|---|---|
| Prototipo (1 device) | $69,67 | ~$0,16 | **~$70** | ~$840 | — |
| **Producción HA, flota 38** (Fargate 1/2 ×2) | $168,70 | $6,15 | **~$175** | ~$2.100 | **~$4,60** |

Fargate de prod canonizado en **1 vCPU/2 GB ×2 = $72/mo** (redundancia entre AZs +
headroom para usuarios concurrentes de Grafana). Alternativa más barata si la
concurrencia es baja: HA-mín 0,5/1 ×2 = $36 → flota ~$139/mo (~$3,65/device).
Crudos: `pricing_full_result.txt`, `pricing_variables_result.txt`, `pricing_prod_result.txt`.

## 2. Runner único reproducible

`scripts/run_benchmarks.py` — orquesta los scripts existentes (no reimplementa),
captura crudos en `docs/benchmarks/<fecha>/raw/` y emite el reporte consolidado
`run_benchmarks_<fecha>.md`. Registra commit + fecha + entorno; idempotente.

```
py scripts/run_benchmarks.py --list                 # lista bloques
py scripts/run_benchmarks.py --group tests cloud     # default (locales)
py scripts/run_benchmarks.py --group sync hardware --pi-host people-counter.local
```

Grupos: `tests` (suites+cobertura+TC sintéticos), `cloud` (e2e/privacidad/TC-18/
costo), `sync` (camsync vía SSH), `hardware` (potencia/soak vía SSH). Marca
**skipped con razón** TC-13 (físico) y TC-01…07 (cenital). Corrida de validación:
**9/9 bloques `tests`+`cloud` → ok, 2 → skipped**.

## 3. Auditoría de seguridad y privacidad (Ley 25.326)

| Control | Resultado | Veredicto |
|---|---|---|
| TLS IoT Core (8883) | TLSv1.2, ECDHE-RSA-AES128-GCM-SHA256, cadena válida (verify 0) | ✅ |
| TLS API (443) | TLSv1.2, cert CN=api.tfg.gasparri.com.ar válido (20-may→3-dic-2026) | ✅ |
| TLS < 1.2 | IoT Core sólo acepta TLS 1.2+ (política AWS); openssl local no pudo bajar a 1.1 para testear rechazo | ✅ (por política) |
| Secretos en claro | repo **0**, logs del device **0**, CloudWatch **0** (AKIA/password/PRIVATE/secret) | ✅ |
| IAM Lambda persist_event | mínimo privilegio: `AWSLambdaBasicExecutionRole` + inline 1 acción `rds-db:connect` a 1 db-user específico; **sin wildcards** | ✅ |
| IAM device (IoT) | scoped al propio Thing (Connect/Subscribe/Receive a su shadow). Publish a `store/*/{counting,wifi_ble,telemetry}` — **nota**: el `store/*` se podría endurecer a per-store | ⚠️ menor |
| Hash de MAC — irreversibilidad | SHA-256 truncado (16 B), one-way, sin tabla reversa (dedup guarda sólo hashes) | ✅ |
| Hash de MAC — rotación intra-día | salt `secrets.token_hex(16)` rotado diario (`reset_daily`); misma MAC → **hash distinto entre días** → inlinkeable | ✅ |

Privacidad por diseño confirmada: ningún dato personal (MAC) se persiste en claro;
el identificador es un hash salteado rotado a diario → **anonimización disociante**
(Ley 25.326). Crudos en `raw/` del runner + sección TC-14.

## 4. Conteo de tests + cobertura

`pytest --cov=src` (regenera la cifra de §6.1). Crudo: `coverage_raw.txt`.

| Métrica | Valor |
|---|---|
| Tests totales | **1079 passed, 2 skipped** (antes 1039) |
| Cobertura global | **81 %** (antes 82 %) |

| Módulo crítico | Cobertura |
|---|---|
| `tracking/counter.py` | 97 % |
| `cloud/persist_event.py` | 98 % |
| `vision/calibration.py` | 92 % |
| `tracking/tracker.py` | 91 % |
| `wifi_ble/dedup.py` (stitching) | 88 % |
| `cloud/query_aggregates.py` | 87 % |
| `mqtt/client.py` | 86 % |
| `mqtt/buffer.py` | 83 % |
| `vision/capture.py` | 61 % |

**Nota honesta**: la global bajó 82→81 % por el código nuevo de `camera_sync` en
`capture.py` (path de hardware/cámaras, no unit-testeable sin sensores) — el
núcleo de lógica sigue 87–98 %.

## 5. Cierres rápidos de TC

| Caso | Resultado | Veredicto |
|---|---|---|
| TC-18 sin autenticación | POST a la API sin SigV4/IAM → **HTTP 403 `Forbidden`** (rutas `/v1/aggregates` y `/pos/transactions` con `AWS_IAM`). Crudo `q_noauth_out.json` | ✅ Cumple |
| TC-11 corte breve (M eventos) | 30 eventos encolados offline → drenados al restablecer: **0 pérdida / 0 duplicado** (`tc11_brief.py`). + evidencia en vivo §A10 (bloqueo nft real → replay) | ✅ Cumple |

### Anexo de método (cierres)
| Item | Comando |
|---|---|
| Costo | `py docs/benchmarks/20260621/cloud_cost.py` |
| Runner | `py scripts/run_benchmarks.py --group tests cloud` |
| Seguridad TLS | `openssl s_client -connect <ep> -tls1_2` |
| Seguridad IAM | `aws iam get-role-policy / aws iot get-policy` (read-only) |
| Hash rotación | `py -c "from src.wifi_ble.hasher import hash_mac; ..."` |
| Cobertura | `pytest --cov=src --cov-report=term-missing:skip-covered` |
| TC-18 no-auth | `py docs/benchmarks/20260621/tc18_noauth.py` |
| TC-11 breve | `py docs/benchmarks/20260621/tc11_brief.py` |
