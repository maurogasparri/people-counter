# run_benchmarks — 20260621

- commit: `1762ac7`
  > **Nota sobre este identificador.** El historial del repositorio fue
  > reescrito con posterioridad a esta campaña, para eliminar material no
  > publicable. El identificador consignado arriba corresponde al historial
  > previo y se conserva **sin modificación**, por tratarse del registro
  > literal de la corrida. Su equivalente en el historial publicado es
  > `c41bb11`.
- inicio: 2026-06-21 22:02:28
- entorno: Windows-11-10.0.26200-SP0 · Python 3.12.6
- grupos: ['cloud', 'tests']
- Pi host: `people-counter.local`

| Bloque | Grupo | Modo | Estado | Resumen |
|---|---|---|---|---|
| coverage | tests | local | ok |  15 files skipped due to complete coverage. 1079 passed, 2 skipped in 42.14s |
| component-suites | tests | local | ok | ........................................................................ [ 62%] .......... |
| TC-09_10-stitching | tests | local | ok |   VEREDICTO TC-10: CUMPLE  RESUMEN: TC-09=PASS TC-10=PASS |
| TC-16-buffer-breve | tests | local | ok | corte breve: 30 eventos encolados offline / unsent=30 / persistidos=True   restablecido: d |
| TC-17-buffer-72h | tests | local | ok | === Control de cap (anti-desborde) ===   inyectados 1500, cap 1000 -> dropeados 500, unsen |
| TC-15-e2e-latency | cloud | local | ok | wifi_ble_events       506    0.278    0.393    0.213    0.586      0  CSV -> docs/benchmar |
| TC-14_11-privacidad-pos | cloud | local | ok |   pos_transactions total: 7122   re-INSERT de transaction_id existente -> filas=0, count 7 |
| TC-13-noauth | cloud | local | ok | TC-13 sin auth -> HTTP 403  body={"message":"Forbidden"} VEREDICTO TC-13 (no-auth): CUMPLE |
| TC-18-corte-energia | tests | manual | ok | ejecutado por separado, fuera de la corrida automatizada (requiere intervención humana) — VEREDICTO TC-18: CUMPLE. Resultado en tc18_powercut_result.txt |
| TC-01..07-conteo | skipped | skip | skipped | conteo controlado — requiere montaje cenital + cruces de personas |

> **Nota.** Este es el manifiesto de la corrida automatizada del 21-06-2026.
> Los resultados definitivos de cada caso son los archivos `*_result.txt` de
> esta carpeta, que en varios casos corresponden a corridas posteriores y más
> completas. Las salidas crudas por bloque que producía esta corrida eran
> copias de esos mismos archivos y se retiraron por duplicadas.
