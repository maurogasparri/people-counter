# Reporte de coverage de tests

Cobertura de la suite de tests unitarios sobre `src/`. Generado con
`pytest-cov` (coverage.py) el **2026-06-16**.

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
| Tests ejecutados | **1008 passed**, 2 skipped |
| Sentencias totales en `src/` | 6740 |
| Sentencias sin cubrir | 1257 |
| **Cobertura total** | **81%** |
| Archivos al 100% | 14 |
| Tiempo de ejecución | ~59 s |
| Plataforma | Python 3.12 (workstation) / target runtime 3.13 (Pi) |

## Cobertura por módulo

Ordenado de mayor a menor cobertura. Los 14 archivos al 100% (la mayoría de
`tests/` espejo de helpers puros: `world_coords`, `hasher`, `kalman`,
`buffer` helpers, `__init__`, etc.) se omiten de la tabla.

| Módulo | Stmts | Cover |
|---|---:|---:|
| `vision/static_suppressor.py` | 52 | 98% |
| `config/hardware.py` | 75 | 97% |
| `status/monitor.py` | 73 | 96% |
| `wifi_ble/fingerprint.py` | 43 | 95% |
| `tracking/counter.py` | 449 | 94% |
| `web/admin_auth.py` | 45 | 93% |
| `vision/calibration.py` | 645 | 92% |
| `config/loader.py` | 384 | 91% |
| `status/health.py` | 68 | 91% |
| `tracking/tracker.py` | 444 | 91% |
| `web/annotate.py` | 184 | 90% |
| `status/led.py` | 89 | 90% |
| `vision/best_frame.py` | 169 | 89% |
| `wifi_ble/dedup.py` | 180 | 88% |
| `cloud/query_aggregates.py` | 301 | 87% |
| `mqtt/client.py` | 220 | 86% |
| `mqtt/buffer.py` | 83 | 83% |
| `cloud/persist_event.py` | 123 | 82% |
| `cloud/ingest_pos_transaction.py` | 125 | 82% |
| `telemetry.py` | 247 | 82% |
| `vision/report.py` | 166 | 81% |
| `web/viewer.py` | 350 | 77% |
| `vision/capture.py` | 277 | 73% |
| `wifi_ble/wifi_probe.py` | 261 | 70% |
| `vision/depth.py` | 235 | 64% |
| `main.py` | 981 | 62% |
| `vision/detect.py` | 230 | 61% |
| `wifi_ble/ble_scan.py` | 101 | 60% |
| **TOTAL** | **6740** | **81%** |

## Interpretación

La cobertura no es uniforme por diseño: **el núcleo algorítmico está alto
(88–98%) y los límites de hardware/I/O están más bajos (60–77%)**.

- **Lógica de negocio bien cubierta (88–98%)**: el `counter` (rescue cascade
  de 3 capas + guards, 94%), el `tracker` (Kalman + state machine + ghost
  pool, 91%), la `calibration` fisheye K-B (92%), el `dedup` con las 4 reglas
  de stitching (88%), `fingerprint` (95%), `static_suppressor` (98%),
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
  en la Pi (commits "validated on hardware"), no en CI.

- **`main.py` (62%)**: es el orquestador. Lo cubierto son los helpers y la
  inicialización; lo no cubierto es el hot loop de captura→detect→track→count
  que solo corre con hardware. Su comportamiento integral se valida con
  `tests/test_main.py` (smoke) + el replay mode (`--replay-dir`) + la
  validación E2E del piloto.

## Conclusión

81% de cobertura total con el **núcleo de decisión (counter/tracker/dedup/
calibration) por encima del 88%**. El gap hasta el 100% son mayoritariamente
los bordes de hardware, cubiertos por validación on-device en lugar de tests
unitarios en CI. Para una flota de dispositivos edge desatendidos es el
trade-off correcto: la red de tests es más densa justo donde un error afecta
el dato entregado al cliente.
