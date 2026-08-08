# Reporte de coverage de tests

Cobertura de la suite de tests unitarios sobre `src/`. Generado con
`pytest-cov` (coverage.py) el **2026-08-08**.

## Cómo regenerarlo

```bash
pip install -e ".[dev]"          # instala pytest + pytest-cov
pytest --cov=src --cov-report=term-missing --cov-report=html
# term-missing → tabla en consola con las líneas sin cubrir
# html        → htmlcov/index.html navegable (gitignored)
```

## Resumen

| Métrica | Valor |
|---|---:|
| Tests ejecutados | **1096 passed**, 7 skipped (1103 total) |
| Sentencias totales en `src/` | 6891 |
| Sentencias sin cubrir | 1303 |
| **Cobertura total** | **81%** |
| Archivos al 100% | 15 |
| Tiempo de ejecución | ~46 s |
| Plataforma | Windows 11, Python 3.12.6, pytest 9.0.2 (workstation) / target runtime 3.13 (Pi) |

## Cobertura por módulo

Ordenado de mayor a menor cobertura. Los 15 archivos al 100% (la mayoría de
`tests/` espejo de helpers puros: `world_coords`, `hasher`, `kalman`,
`buffer` helpers, `__init__`, etc.) se omiten de la tabla.

| Módulo | Stmts | Cover |
|---|---:|---:|
| `vision/static_suppressor.py` | 52 | 98% |
| `cloud/persist_event.py` | 123 | 98% |
| `config/hardware.py` | 75 | 97% |
| `tracking/counter.py` | 449 | 97% |
| `status/monitor.py` | 74 | 96% |
| `wifi_ble/fingerprint.py` | 43 | 95% |
| `web/admin_auth.py` | 45 | 93% |
| `vision/calibration.py` | 645 | 92% |
| `config/loader.py` | 384 | 91% |
| `status/health.py` | 68 | 91% |
| `tracking/tracker.py` | 450 | 91% |
| `web/annotate.py` | 184 | 90% |
| `status/led.py` | 89 | 90% |
| `vision/best_frame.py` | 169 | 89% |
| `wifi_ble/dedup.py` | 206 | 82% * |
| `cloud/query_aggregates.py` | 301 | 87% |
| `mqtt/client.py` | 220 | 86% |
| `mqtt/buffer.py` | 83 | 83% |
| `telemetry.py` | 247 | 82% |
| `vision/report.py` | 166 | 81% |
| `web/viewer.py` | 350 | 77% |
| `wifi_ble/wifi_probe.py` | 261 | 70% |
| `vision/depth.py` | 235 | 64% |
| `main.py` | 987 | 62% |
| `vision/capture.py` | 371 | 61% |
| `vision/detect.py` | 230 | 61% |
| `wifi_ble/ble_scan.py` | 101 | 60% |
| **TOTAL** | **6891** | **81%** |

> **Pasada de hardening de tests (2026-06-17)**. Se cerraron los gaps de
> mayor riesgo entre el núcleo de decisión y las Lambdas:
> - **`counter` 94%→97%**: los guards anti-FP del death-emit (capa 3 del
>   rescue cascade: `min_real_inside_frames`, `min_count_height_m`,
>   `min_count_confidence`) ahora se ejercitan en aislamiento. Los tests e2e
>   previos assertaban el outcome (`total_in == 0`) pero un guard anterior
>   (confidence) cortaba primero → las líneas del guard nombrado nunca
>   corrían (false confidence). Los nuevos tests llaman `_emit_on_death` con
>   un snap controlado: pasa todos los guards menos el target, con control que
>   confirma el aislamiento.
> - **`persist_event` 82%→98%** y **`ingest_pos_transaction` 82%→100%**: paths
>   de resiliencia del Lambda caliente (reconexión de conexión stale vía
>   health-check `SELECT 1`, swallow del close fallido, selección de
>   `sslmode`/`sslrootcert`) + ramas de validación (event_ts de tipo
>   inválido, transacción no-dict, batch vacío, error transitorio vs error de
>   datos). Son Python puro, 100% mockeable — exactamente las rutas de "qué
>   pasa cuando RDS hipa".
>
> En la misma pasada se corrigió una **race latente** en
> `tests/web/test_viewer.py::test_login_and_power_flow`: el handler `/reboot`
> responde 200 *antes* de disparar la acción (en el thread del server), y el
> test assertaba el side-effect al toque. Pasaba aislado y fallaba con la
> suite completa según el scheduling; ahora espera el efecto con timeout.
>
> El total agregado se movió poco con esa pasada; en la medición de entonces
> (2026-06-26) era **81%** — el código de runtime sumado después (p. ej.
> `camera_sync` en `capture.py`, que pasó de 73% a 61% al crecer ~90
> sentencias que no se ejercitan sin hardware) diluyó el agregado. El
> denominador está dominado por los bordes de hardware + `main.py` que no se
> persiguen por diseño; el valor está en los módulos de mayor riesgo a 97-98%,
> no en el número agregado.

\* `wifi_ble/dedup.py` incluye el ajuste de permisos del archivo SQLite,
que es no-op fuera de POSIX: medido sobre Linux su cobertura es mayor. La
cifra de la tabla corresponde a la corrida sobre la máquina de desarrollo.

## Interpretación

La cobertura no es uniforme por diseño: **el núcleo algorítmico está alto
(82–98%) y los límites de hardware/I/O están más bajos (60–77%)**.

- **Lógica de negocio bien cubierta (82–98%)**: el `counter` (rescue cascade
  de 3 capas + guards, 97%), el `tracker` (Kalman + state machine + ghost
  pool, 91%), la `calibration` fisheye K-B (92%), el `dedup` con las 4 reglas
  de stitching (82%), `fingerprint` (95%), `static_suppressor` (98%),
  `config/loader` strict (91%) y los probes de salud (`health`/`monitor`/`led`
  90–96%). Es el código donde un bug se traduce en counts incorrectos — y es
  el que tiene la red de tests más densa.

- **Límites de hardware sub-cubiertos (60–77%) — esperado**: los módulos que
  hablan directo con periféricos del device **no se pueden ejercitar en la
  workstation** (sin cámaras CSI, sin Hailo, sin nexmon/monitor mode, sin
  BlueZ/D-Bus):
  - `vision/detect.py` (61%) — runtime VStream del Hailo (`hailo_platform`).
  - `vision/capture.py` (73%) — picamera2 dual-cam.
  - `vision/depth.py` (64%) — los paths pesados de SGBM/WLS sobre frames reales.
  - `wifi_ble/wifi_probe.py` (70%) — captura 802.11 en monitor mode.
  - `wifi_ble/ble_scan.py` (60%) — escaneo pasivo BLE vía bleak.

  En todos, lo que queda sin cubrir son las ramas que requieren el
  dispositivo físico; la lógica pura aledaña (parsing, transforms, filtros)
  sí está testeada. Estas rutas se validan con los smoke tests *on-hardware*
  en la Pi (commits "validated on hardware"), no con la ejecución local de la
  suite.

- **`main.py` (62%)**: es el orquestador. Lo cubierto son los helpers y la
  inicialización; lo no cubierto es el hot loop de captura→detect→track→count
  que solo corre con hardware. Su comportamiento integral se valida con
  `tests/test_main.py` (smoke) + el replay mode (`--replay-dir`) + la
  validación E2E en las instalaciones.

## Conclusión

81% de cobertura total con el **núcleo de decisión (counter/tracker/dedup/
calibration) por encima del 82%** (counter y las Lambdas de ingesta en
97-98% tras la pasada de hardening). El gap hasta el 100% son mayoritariamente
los bordes de hardware, cubiertos por validación on-device en lugar de tests
unitarios en la ejecución local de la suite. Para una flota de dispositivos edge desatendidos es el
trade-off correcto: la red de tests es más densa justo donde un error afecta
el dato entregado al cliente.
