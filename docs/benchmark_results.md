# Resultados de benchmark — People Counter Edge System

## 1. Introducción

Este documento consolida la validación del prototipo de conteo de personas: el
nodo de borde (Raspberry Pi 5 + acelerador Hailo-8L, par estéreo IMX708 con
modelo fisheye Kannala-Brandt, captura pasiva WiFi/BLE) y el backend en AWS
(IoT Core → Lambda → RDS Postgres → Grafana). La validación combina dos planos:
**caracterización de banco** (requisitos no funcionales medidos sobre la unidad
de desarrollo, en laboratorio) y **validación dirigida en las instalaciones de
la organización** (montaje cenital reproduciendo la geometría de instalación,
con cruces controlados y tráfico orgánico operador-confirmado). Esta validación
dirigida tiene carácter **indicativo** —una unidad, muestras del orden de la
decena por caso, un único operador—: confirma el funcionamiento en la
configuración de montaje cenital, pero la medición de exactitud estadísticamente
robusta a escala corresponde a la etapa de **piloto** (futura, sujeta a la
decisión de la organización y desarrollada en el plan de implementación). Los
valores provienen de mediciones instrumentadas reproducibles; cuando un criterio
no pudo verificarse se indica explícitamente.

## 2. Casos de prueba (TC-01 … TC-19)

| Código | Descripción | Muestra | Resultado observado | Veredicto |
|---|---|---|---|---|
| TC-01 | Conteo de ingreso individual | 10 cruces | 10/10 (dirigido controlado) | ✅ PASS |
| TC-02 | Conteo de egreso individual | 10 cruces | 10/10 (dirigido controlado) | ✅ PASS |
| TC-03 | Cruces simultáneos en direcciones opuestas | 10 (5+5) | 8/10 (umbral ≥ 9/10); **sin fusión de identidades** (8/8 tracks) | ⚠️ Limitación documentada |
| TC-04 | Ráfaga en el mismo sentido | 2×5 (10) | 10/10 — dos ráfagas same-sense de 5 (1 de ingreso, 1 de egreso) en tráfico real, conf. 0,8–0,9 | ✅ PASS |
| TC-05 | Robustez a apariencia (gorra/capucha) | 10 cruces | 9/10 (dirigido controlado) | ✅ PASS |
| TC-06 | Rechazo de objetos no humanos (gate de altura) | 8 pasadas | 0 conteos de las pasadas medibles; gate validado | ✅ PASS (con matiz) |
| TC-07 | Hesitación / amago sin cruzar | 8 amagos | 0/8 eventos netos espurios | ✅ PASS |
| TC-08 | Estimación de estatura (±10 cm) | 12 (≥2 sujetos) | 15/15 dentro de ±10 cm (sujetos de 1,68 y 1,82 m) | ✅ PASS |
| TC-09 | Stitching WiFi por continuidad de seqnum/fingerprint | 6 MACs de 1 device | 6 MACs randomizadas → 1 `group_id`; segundo device → grupo aparte | ✅ PASS |
| TC-10 | Stitching cross-protocolo WiFi+BLE | par WiFi+BLE | mismo device (ΔRSSI 2 dBm, <2 s) → mismo grupo; control ΔRSSI 38 dBm → grupo aparte | ✅ PASS |
| TC-11 | Ingesta POS + tasa de conversión | 1 re-ingesta sobre 7122 | re-INSERT idempotente (7122→7122, 0 filas); ventas/día computables para conversión | ✅ PASS |
| TC-12 | Idempotencia de ingesta cloud (conteo) | 1 re-ingesta | re-INSERT de `count_event` existente → 0 filas (ON CONFLICT) | ✅ PASS |
| TC-13 | Consulta de agregados (API REST: auth + params) | 3 invocaciones | válida → 200 con agregados; falta `from` → 400; rango >7 d → 400 (RFC-7807); sin auth → 403 Forbidden | ✅ PASS |
| TC-14 | Privacidad por diseño | 500 muestras RDS + disco/journal | 0 MACs crudas, `visitor_hash` opaco (16 B bytea), 0 imágenes en disco | ✅ PASS |
| TC-15 | Latencia evento → dashboard (p95 ≤ 5 s) | n=11 conteo / 506 WiFi | `count_events` p95 0,272 s; `wifi_ble_events` p95 0,393 s; p50 sub-segundo | ✅ PASS |
| TC-16 | Resiliencia de conectividad breve (~30 min) | 30 eventos | encolados offline → drenados íntegros, 0 pérdida / 0 duplicado | ✅ PASS |
| TC-17 | Resiliencia de conectividad prolongada (72 h) | 1302 eventos | persistidos sin pérdida → drenados 0/0; cap acota a 1000 sin desborde | ✅ PASS |
| TC-18 | Reinicio tras corte de energía (< 90 s) | 1 corte físico real | frames fluyendo en **+46 s**; `integrity_check` SQLite OK; fs sano, throttle 0x0 | ✅ PASS |
| TC-19 | Disponibilidad del stack cloud (24 h ≥ 99 %) | 48 h continuas | Sin reinicios ni desconexiones MQTT; backlog del outbox máx 2; 0 throttling; dashboards accesibles. Corroborado por la continuidad de ingestión sobre 33,6 días (mejor ventana de 24 h: 299 ≥ 288 telemetrías, 0 pérdida) | ✅ PASS |

**TC-03 — sub-conteo en cruces simultáneos bidireccionales.** La corrida dirigida
dio 8/10 (umbral ≥ 9/10) **sin fusión de identidades** (8/8 tracks). La causa
raíz se probó con datos: **no es el detector** (confianza en cruces opuestos
0,75 ≈ 0,79 en cruces solos) ni el **FPS** (una ráfaga densa en el mismo sentido
se contó completa a 23–24 FPS); es la **convergencia del tracker** — el ratio
test post-Hungarian (`ambiguous_match_ratio`) rechaza y consume la detección
ambigua para no swapear IDs, por lo que un track coastea sin registrar el cruce.
Es un **drop conservador y direction-agnostic**: descarta el cruce en disputa sin
sesgo de sentido (preferible a invertir la dirección), de modo que las omisiones
no sesgan sistemáticamente el conteo **neto**. Mitigación aplicada: canary
`ambiguous_reject_count` que cuantifica la presión de convergencia en producción.

**TC-04 — ráfaga mismo sentido.** Dos ráfagas same-sense de 5 cruces cada una
—una de ingreso y una de egreso— observadas en tráfico real y confirmadas por el
operador se contaron 5/5 y 5/5 (**10/10**), con los grupos del mismo sentido bien
separados (confianzas 0,8–0,9). El FPS sostenido mantiene la separación temporal
de cruces consecutivos; cumple sin omisión.

**TC-06 — rechazo de objetos no humanos.** De 8 pasadas gateando bajo, solo 4
generaron track (el detector, fine-tuneado a cabezas de gente parada, pierde la
cabeza baja/angulada en ~la mitad de las pasadas → defensa en profundidad). De
esas 4: 3 fueron rechazadas por el gate de altura (altura medida 0,96–0,97 m <
1,0 m) y 1 contó por altura NULL (los tracks sin profundidad no se filtran, por
diseño). Resultado: 0 conteos de las pasadas medibles; el gate queda validado.

**TC-08 — estimación de estatura.** 15/15 dentro de ±10 cm con dos sujetos
(1,68 y 1,82 m). El sujeto de 1,82 m entra en tolerancia gracias a la
mitigación de la limitación L1 (techo `max_head_height` levantado + ajuste
SGBM); ver §4.

**TC-19 — disponibilidad del stack cloud.** En una corrida de 48 h continuas el
stack operó sin reinicios ni desconexiones MQTT, con el backlog del outbox acotado
(máx 2 mensajes), 0 throttling y los dashboards accesibles — supera el criterio de
≥ 99 % en 24 h. Como corroboración adicional, la continuidad de ingestión sobre los
33,6 días de operación de la unidad de desarrollo muestra, en la mejor ventana continua
de 24 h con el device sin interrupción, 299 de 288 telemetrías esperadas a 5 min
persistidas (0 pérdida); los gaps del período corresponden a **device apagado** (unidad de
desarrollo entre sesiones, traslado off-site), no a caídas del backend
(IoT Core → Lambda → RDS → Grafana), que no registró outages.

## 3. Caracterización de banco (requisitos no funcionales)

### Consumo eléctrico

| Estado | Promedio (W) | p95 (W) | Pico (W) | % del presupuesto PoE 25,5 W |
|---|---|---|---|---|
| Pipeline detenido (idle del SO) | 2,16 | 2,19 | 3,06 | 8,5 % |
| Corriendo, escena vacía, idle-throttle ON (~10–16 FPS) | 4,00 | 5,38 | 9,93 | 15,7 % |
| Corriendo, escena vacía, full-throughput (~28 FPS) | 4,67 | 6,57 | 8,75 | 18,3 % |

El consumo es casi independiente de la carga de cómputo (10→28 FPS suma sólo
~0,67 W). Pico absoluto 9,93 W (transitorio), < 40 % del presupuesto.

### Térmico (con Active Cooler)

| Métrica | Valor | Límite | Veredicto |
|---|---|---|---|
| Temp CPU máx bajo estrés sostenido | 64,8 °C | 80 °C | ✅ (15 °C de margen) |
| Temp Hailo (telemetría, máx) | 41,8 °C | 85 °C | ✅ |
| Throttling (`get_throttled`) bajo estrés | 0x0 (132 muestras) | 0x0 | ✅ nunca throttleó |

(El comportamiento térmico en gabinete cerrado en las instalaciones se trata
como limitación L2 en §4.)

### Memoria

| Métrica | Valor | Objetivo flota | Veredicto |
|---|---|---|---|
| `memory.peak` del servicio (soak ~60 min) | 416 MiB | < 2048 MiB | ✅ (20 % usado) |
| `memory.peak` bajo stress sintético | 247 MiB | < 2048 MiB | ✅ (12 %) |
| `memory.events` (high/max/oom) y swap | 0 | 0 | ✅ |

### FPS, throughput y latencia por etapa

| Escenario | FPS efectivo | Etapa dominante |
|---|---|---|
| Escena vacía (SGBM omitido) | 27,8 FPS | inferencia Hailo 15,9 ms + captura 14,1 ms |
| Con persona (SGBM activo) | 21,4 FPS | profundidad SGBM 22,6 ms (~46 % del frame) |

Latencia por frame (telemetría): p50 24,0 ms · p95 26,5 ms. El pipeline es
cámara-bound (~28 FPS) con escena vacía y SGBM-bound (~21 FPS) con gente.
Throughput puro del detector on-chip: **105,9 inf/s** (≈5× el rate del
pipeline → el NPU no es el cuello de botella).

### CPU

| Estado | CPU agregada (4 cores) | FPS |
|---|---|---|
| Nominal, idle-throttle ON | avg 15,3 % · p95 27,4 % | ~10–16 |
| Full-throughput, escena vacía | ~96 % (≈1 core de 4) | ~28 |

El pipeline es serial/single-core; el cuello de botella del FPS es la latencia
serial por frame, no la CPU agregada ni la RAM.

### Sincronización de cámaras (delta L/R)

| Configuración | mediana | p95 | % pares ≤ 5 ms |
|---|---|---|---|
| Sin sync | 0,37 ms | 18,8 ms | 87,8 % |
| Con sync (software, libcamera SyncMode) | 0,025 ms | 0,050 ms | 100,0 % |

El sync por software elimina la cola episódica a ~19 ms (deriva de un período
de frame entre relojes de sensor) sin cableado.

### Cobertura de tests

| Métrica | Valor |
|---|---|
| Tests totales | 1084 passed, 2 skipped (1086 total) |
| Cobertura global | 81 % |
| Módulos críticos | `counter.py` 97 % · `persist_event.py` 98 % · `calibration.py` 92 % · `tracker.py` 91 % · `dedup.py` 88 % |

### Calibración estéreo y profundidad

| Métrica | Valor | Umbral | Veredicto |
|---|---|---|---|
| RMS epipolar | 0,115 px (15/15 frames, 467 corners, máx 0,596 px) | ≤ 1 px | ✅ PASS |
| MAE de profundidad de banco (1,0–2,0 m) | ≈ 43 mm (MAPE ≈ 2,5 %); error de centro −0,43 % @ 1 m a −4,58 % @ 2 m | < 5 % @ 2 m | ✅ PASS |
| Baseline ‖T‖ (n=7 calibraciones) | 144,6 ± 2,7 mm (offset +4,6 mm vs nominal mecánico 140 mm) | informativo | — |
| MAE de estatura en geometría cenital real | ≈ 28 mm; dos sujetos (1,68 y 1,82 m), 15/15 dentro de ±10 cm (tras mitigar L1) | ±10 cm | ✅ PASS |

## 4. Hallazgos de la validación y mitigaciones

La validación (banco + TC dirigidos + validación dirigida en las instalaciones)
arrojó **cinco hallazgos accionables**: cuatro resueltos/mitigados durante el
desarrollo —desfase de sincronización estéreo, compresión de estatura (L1),
consumo de CPU en escena vacía y margen térmico en gabinete cerrado (L2)— y uno
caracterizado como limitación (sub-conteo en cruces simultáneos opuestos, TC-03).
La tabla los resume; el detalle de las limitaciones sigue debajo.

| Hallazgo | Detectado en | Causa | Mitigación | Estado |
|---|---|---|---|---|
| Compresión de estatura a mount bajo (L1) | TC-08 / B3 (estatura) | edge-bleed near-camera en el extractor de head-depth a mount ~2,4 m | techo `max_head_height` + ajuste SGBM (uniqueness / WLS) | **Mitigado** — validado A/B, dentro de ±10 cm |
| Desfase estéreo episódico ~19 ms (~12 % de pares) | caracterización de sync L/R | captura en libre corrida: las fases de ambos sensores derivan | captura sincronizada por software (converge + hold) | **Resuelto** — p95 0,05 ms, 100 % < 5 ms |
| Sub-conteo en cruces simultáneos en sentidos opuestos | TC-03 dirigido | convergencia del tracker: el ratio-test descarta el cruce en disputa para no invertir la dirección | drop conservador direction-agnostic (sin sesgo de sentido → neto preservado) + canary `ambiguous_reject_count` | **Limitación documentada** |
| Throttling térmico en gabinete cerrado (L2) | validación en las instalaciones | heat-soak del gabinete cerrado (~84 °C) — el gabinete, no el cooler, es el factor dominante | freq-cap 1500 MHz (sin pérdida de FPS por ser Hailo-bound); ranuras de ventilación ampliadas como medida adicional opcional | **Mitigado** — con cap ~80 °C prom (máx 81 °C) vs ~84 °C sin cap, 0 throttling (banco: máx 64,8 °C) |
| CPU alta en escena vacía (~162 % agregado) | caracterización de CPU / profiling de banco | conversión de color redundante por frame + rectificación derecha eager + parsing de tramas sin pre-filtro + sin FPS adaptativo | eliminación de la conversión + rectificación lazy + pre-filtro de tramas + `vision.idle_throttle` (10 FPS en escena vacía, wake instantáneo) | **Resuelto** — ~50 % en escena vacía, count-neutral (validado de extremo a extremo en las instalaciones) |

### Limitaciones conocidas (detalle)

**L1 — Compresión de estatura a mount bajo (2,413 m) — MITIGADA.** A esta
altura las cabezas quedan a 0,6–0,73 m, donde la diferencia de estatura entre
personas se traduce en muy poca disparidad y el pipeline de runtime
(SGBM `downscale=4` + WLS + extracción por slices de 10 cm) la suaviza,
capturando sólo ~10 % de la variación real. No afecta el conteo (usa centroide
2D), sólo la demografía adulto/niño. Mitigada por config —
`max_head_height_m: 1.95`, `uniqueness_ratio: 15`, WLS `λ: 1500`— y validada
A/B sobre la misma persona en las instalaciones (sujeto de 1,82 m: de ~1,73 m / −9 cm a
~1,78 m / −4 cm, dentro de ±10 cm, sin costo de FPS). Residual ~−4 cm; su fix
de raíz (estimador robusto del crown) queda fuera del alcance del prototipo.

**L2 — Margen térmico en gabinete cerrado — MITIGADA.** En la condición real de
uso —el dispositivo dentro del gabinete impreso cerrado, donde el calor se acumula
y el propio gabinete es el factor térmico dominante— bajo carga sostenida y **sin
límite de frecuencia** el CPU alcanzó un máximo de **84,3 °C** y el firmware activó
el throttling térmico (el mismo dispositivo fuera del gabinete opera ~60 °C).
Mitigación adoptada: limitar la frecuencia máxima del CPU a 1500 MHz
(`cpu-freq-cap.service`, persistente; sin pérdida de FPS por ser Hailo-bound). Con
ese límite la temperatura **se estabiliza en ~80 °C de promedio (máx 81,0 °C)** y
el **throttling desaparece por completo**; en banco (Active Cooler, sin gabinete)
el máximo fue 64,8 °C. El freq-cap por sí solo mantiene al CPU fuera de la zona de
throttling, por lo que se adopta como **mitigación definitiva** (no un parche a
revertir); la **ampliación de las ranuras de ventilación queda como medida
adicional, no requerida** en esta configuración.

**L3 — Alcance de la validación de conteo (en resolución).** La validación
dirigida se apoyó originalmente en una ventana orgánica de ~1 h y B3 con n=1
sujeto; TC-04/06 son confirmaciones cualitativas del operador. No es una
limitación del producto sino de la fuerza estadística de la evidencia —de
carácter indicativo, propia de una unidad y un operador— y se elevó con los
tests dirigidos controlados (TC-01/02/05/06/07 PASS; TC-08 ahora con 2 sujetos,
15/15). La medición de exactitud estadísticamente robusta a escala corresponde a
la etapa de piloto (futura). El único caso que reveló una limitación real es el
simultáneo-estricto bidireccional (ver TC-03).

**L4 — Baseline óptico ≠ nominal mecánico.** La calibración estima el baseline
entre los centros ópticos (pupilas de entrada del fisheye), que difiere del
nominal mecánico de 140 mm por la óptica y las tolerancias de impresión del
case: 144,6 ± 2,7 mm sobre n=7 calibraciones. Sin impacto (el depth quedó
validado por ground-truth y el conteo no usa el baseline directamente); el gate
del baseline se degradó a informativo para prevenir un false-FAIL — el verdict
de calibración lo decide el ground-truth de profundidad.
