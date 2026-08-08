# Evidencia de validación

Evidencia primaria de los resultados publicados en
[`docs/benchmark_results.md`](../docs/benchmark_results.md). **Para leer los
resultados no hace falta abrir nada de acá.**

Cada archivo cumple uno de tres roles: **ejecuta** una prueba, **es la salida**
de una prueba, o **procesa** los datos crudos de una medición. No hay nada más.
Los nombres empiezan por el caso al que pertenecen, de modo que el listado los
agrupa solo, y **cada caso tiene una única salida**.

Los casos TC-01 a TC-08 son cruces de personas bajo la cámara: su resultado sale
de los eventos persistidos en la base de datos. Lo que esta carpeta aporta de esa
campaña es el **banco que fija los criterios**, la **traza por repetición** del
día y las **mediciones de estatura**; ver la sección siguiente.

## Campaña dirigida del 25 de junio de 2026

| Archivo | Qué es |
|---|---|
| `tc_controlled.py` | Banco de la campaña: define el criterio de aceptación de cada caso y evalúa la ventana contra la base de datos |
| `tc_audit.md` | Reconstrucción de las 656 visitas a la zona de conteo de la jornada: entradas, cruces con su lado y balance, veredicto de salida, muertes, adopciones y rechazos por guarda |
| `tc_trace.csv` | La misma traza, una fila por visita, para inspección |
| `tc03_simulacion_result.txt` | Salida del banco `scripts/analysis/simulate_associator.py`: simetría por reflexión, prueba pareada y superficie de decisión. **Caracteriza el algoritmo, no el sistema completo**, con modelo de ruido no calibrado. Incluye las dos parametrizaciones: la nominal y una **adversa** (separación 10 px, p_miss 0,25), que es donde puede ponerse a prueba la simetría. Es la segunda evidencia de TC-03, junto a `tc03_result.txt` |
| `height_mae.csv` | Las quince mediciones de estatura de TC-08: instante del cruce, sujeto, estatura real y estimada. **Cada fila es un cruce distinto**; los valores estimados que se repiten son mediciones independientes que coincidieron. El instante se recuperó del registro persistido para que cada fila sea verificable por separado |
| `height_mae.py` | Guion que recogió esas mediciones desde la base de datos |
| `count_session.py` | Guion de conteo por sesión usado durante la campaña |

La auditoría y la traza se **regeneraron el 2026-08-08** con
`scripts/analysis/audit_directed_trials.py` sobre el registro de aplicación del
propio dispositivo, que conserva la jornada completa. El guion y la fuente son
los de la campaña; la salida original no se conservó. La auditoría lo declara en
su encabezado.

Las direcciones de la red donde estuvo instalado el equipo fueron sustituidas
por `<ip-del-dispositivo>`.

## Casos de prueba

| Caso | Ejecuta | Salida |
|---|---|---|
| TC-09 / TC-10 — stitching WiFi y entre protocolos | `tc09_10_stitching.py` | `tc09_10_result.txt` |
| TC-11 — tasa de conversión de extremo a extremo | `tc11_conversion_rate.py` | `tc11_conversion_result.txt` |
| TC-12 — idempotencia de la ingesta en la nube | `tc12_idempotency.py` | `tc12_idempotency_result.txt` |
| TC-13 — control de acceso y validación de parámetros | `tc13_api_checks.py` | `tc13_result.txt` |
| TC-14 — privacidad por diseño | — | `tc14_result.txt` — barrido en la nube y auditoría del disco del dispositivo |
| TC-15 — latencia de extremo a extremo por invocación | `tc15_latency_by_invocation.py` | `tc15_latency_by_invocation_result.txt` |
| TC-16 — corte breve de conectividad | `tc16_brief.py` | `tc16_result.txt` |
| TC-17 — corte prolongado (volumen de 72 h) | `tc17_buffer_72h.py` | `tc17_result.txt` |
| TC-18 — reinicio tras corte de energía real | — | `tc18_powercut_result.txt` |
| TC-19 — disponibilidad del stack en la nube | `tc19_cloud_availability.py` · `tc19_alarm_reconstruction.py` · `tc19_dashboard_reachability.py` | `tc19_result.txt` — los tres bloques en un solo archivo |

## Caracterización de banco

| Medición | Datos crudos | Procesa |
|---|---|---|
| Consumo eléctrico | `power_idle.csv` · `power_session.csv` · `power_fullthroughput.csv` | `analyze_bench.py` |
| Térmico | `stress_monitor.csv` · `soak_system.csv` · `thermal_deploy_uncapped.csv` · `thermal_deploy_capped_1500mhz.csv` | `analyze_bench.py` → `analysis_summary.txt` |
| Memoria | `stress_monitor.csv` · `soak_system.csv` | `memory_working_set.py` → `memory_working_set_result.txt` |
| FPS, throughput y latencia por etapa | `profile_empty.log` · `profile_empty_perframe.csv` | `parse_profile.py` → `profile_empty_summary.txt` |
| Sincronización de cámaras (delta L/R) | `camsync_sin_sync.csv` · `camsync_con_sync.csv` | `camsync_sin_sync.py` · `camsync_con_sync.py` |
| Cobertura de tests | — | `test_suite_coverage.txt` |
| Foco (19-06) | — | `focus_report.txt` — transcripción del informe del asistente |
| Calibración desplegada (22-06) | `calib_run.log` — registro de la corrida que la produjo | `calib_gt.json` — su verificación contra distancia medida |

La calibración desplegada en sí —`calibration.npz`— se distribuye
como adjunto de la versión etiquetada, no versionada en el repositorio.
SHA-256 `4cb38d0809060873ca014fadca4f3034f3013ec30ec5e9e37529290e67e37d96`.

`run_benchmarks.md` es el manifiesto de la corrida automatizada del
21-06-2026, conservado como registro de qué se ejecutó ese día.

## Demostración funcional del 6 de agosto de 2026

`demo_crossings.py` extrae de la base los ocho cruces registrados durante la
grabación del video de demostración, y `demo_crossings_result.txt` es su
salida: encabeza con la consulta literal, tabula cada evento con su instante,
sentido, estatura, confianza y latencia, y cierra contrastando las magnitudes
contra las que cita el trabajo.

**No es un caso de prueba y no lleva identificador de caso.** Con dos cruces
por condición la muestra no alcanza ningún criterio de aceptación —el de
conteo individual exige 9 de 10—, de modo que documenta una **demostración**
del funcionamiento de la cadena de extremo a extremo, no una verificación.

## Reproducir

Desde la raíz del repositorio:

```
py validation/tc09_10_stitching.py     # autónomo, base temporal
py validation/tc17_buffer_72h.py       # autónomo, base temporal
py validation/tc12_idempotency.py      # requiere credenciales de AWS
```

Los de componente —stitching y buffer— corren sobre bases temporales y no tocan
nada persistente. Los que consultan la nube necesitan credenciales con acceso a
la base de datos y a CloudWatch.
