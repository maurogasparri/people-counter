# Auditoría de la traza — 2026-06-25

> Evidencia documental de ensayos ya reportados. La traza registra lo
> que el sistema decidió, no lo que ocurrió frente a la cámara: **no
> tiene verdad de referencia** y de ella no se derivan tasas.

## Cobertura y contexto

- Fuente: `app.log` del dispositivo (registro de aplicación en disco, no journald),
  filtrado a las 3 015 líneas `TRACKDBG` del 2026-06-25.
- **Regenerado el 2026-08-08** con `scripts/analysis/audit_directed_trials.py` sobre
  ese mismo registro. El guion y la fuente son los de la campaña; la salida original
  del 25-06 no se conservó.
- Primera línea TRACKDBG: 08:31:16
- Última línea TRACKDBG: 19:06:30
- Reinicios del servicio en la jornada: 0
- Counting zone: x 344–1004, y 82–322; línea y=202
- Visitas a la zona reconstruidas: 656
  - con conteo: 515 · sin conteo: 102 · suprimidas por guarda: 39
- Líneas TRACKDBG por tipo: entry=656, cross=546, exit=654, death=731, exit_kalman_skipped=27, entry_kalman_skipped=378, ghost_adopted=11, exit_thin_evidence_skipped=9, exit_short_height_skipped=3

- **Canary `ambiguous_reject_count`**: sólo reporta desde las 12:01:31 de esa jornada. Cualquier rechazo del ratio-test anterior a esa hora no quedó registrado en ninguna fuente.

## Pares convergentes en sentidos opuestos

Visitas solapadas en el tiempo, de lados opuestos de la línea y con separación mínima observada ≤ 250 px — la condición que somete al asociador a desambiguar.

| Hora | tid A | tid B | separación mín. | resultado A | resultado B | canary de ambigüedad activo |
|---|---:|---:|---:|---|---|---|
| 08:34:50 | 5 | 6 | 94 px | contada | sin conteo | NO |
| 08:35:21 | 11 | 13 | 18 px | sin conteo | sin conteo | NO |
| 08:35:32 | 14 | 15 | 83 px | sin conteo | sin conteo | NO |
| 08:54:46 | 64 | 65 | 22 px | suprimida | sin conteo | NO |
| 09:17:32 | 139 | 140 | 41 px | contada | sin conteo | NO |
| 13:13:11 | 111 | 114 | 246 px | contada | contada | sí |
| 13:13:26 | 116 | 117 | 129 px | suprimida | sin conteo | sí |
| 13:15:48 | 125 | 126 | 3 px | contada | sin conteo | sí |
| 17:36:26 | 593 | 594 | 223 px | contada | sin conteo | sí |
| 18:53:32 | 671 | 672 | 4 px | contada | sin conteo | sí |

De 10 pares convergentes, 5 ocurrieron con el canary `ambiguous_reject_count` ya reportando y 5 antes de que empezara a hacerlo.

## Ráfagas de actividad (≥ 4 visitas, hueco ≤ 10 s)

Una ráfaga es candidata a ser un ensayo dirigido. **La traza no contiene marca de ensayo**: la correspondencia con un código TC concreto no puede establecerse desde el registro.

### Ráfaga 1 — 08:34:50–08:35:23 (7 visitas, 3 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 08:34:50 | 5 | +1 | sí | — | -1 | egress | 11 | contada |  |
| 08:34:51 | 6 | -1 | sí | — | 0 | None | 5 | sin conteo | no registró cruce de línea |
| 08:34:56 | 7 | -1 | sí | ingress(+1) | 1 | ingress | 5 | suprimida | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 08:35:04 | 10 | +1 | sí | egress(-1) | -1 | egress | 18 | contada |  |
| 08:35:12 | 11 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |
| 08:35:21 | 11 | +1 | sí | — | 0 | None | 12 | sin conteo | no registró cruce de línea |
| 08:35:22 | 13 | -1 | sí | — | 0 | None | 6 | sin conteo | no registró cruce de línea |

### Ráfaga 2 — 08:35:32–08:35:50 (5 visitas, 1 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 08:35:32 | 14 | -1 | sí | ingress(+1) → egress(+0) | 0 | None | 32 | sin conteo | balance neto cero (ida y vuelta) |
| 08:35:33 | 15 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 08:35:39 | 15 | +1 | sí | — | — | — | — | sin conteo | no registró cruce de línea |
| 08:35:48 | 17 | -1 | sí | ingress(+1) | 1 | ingress | 9 | contada |  |
| 08:35:49 | 17 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |

### Ráfaga 3 — 09:01:58–09:02:33 (13 visitas, 2 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:01:58 | 89 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 09:02:01 | 91 | -1 | sí | ingress(+1) | 1 | ingress | 7 | contada |  |
| 09:02:03 | 93 | +1 | sí | — | 0 | None | 5 | sin conteo | no registró cruce de línea |
| 09:02:04 | 93 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 09:02:05 | 93 | +1 | sí | — | 0 | None | 43 | sin conteo | no registró cruce de línea |
| 09:02:11 | 93 | +1 | sí | — | 0 | None | 10 | sin conteo | no registró cruce de línea |
| 09:02:11 | 93 | +1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 09:02:12 | 93 | +1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 09:02:12 | 93 | +1 | sí | — | 0 | None | 4 | sin conteo | no registró cruce de línea |
| 09:02:12 | 93 | +1 | sí | — | 0 | None | 3 | sin conteo | no registró cruce de línea |
| 09:02:14 | 93 | +1 | sí | — | 0 | None | 12 | sin conteo | no registró cruce de línea |
| 09:02:18 | 93 | +1 | sí | — | 0 | None | 6 | sin conteo | no registró cruce de línea |
| 09:02:22 | 93 | +1 | sí | — | 0 | None | 130 | sin conteo | no registró cruce de línea |

### Ráfaga 4 — 09:02:33–09:03:13 (7 visitas, 1 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:02:33 | 93 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 09:02:33 | 93 | +1 | sí | — | 0 | None | 3 | sin conteo | no registró cruce de línea |
| 09:02:34 | 93 | +1 | sí | — | 0 | None | 4 | sin conteo | no registró cruce de línea |
| 09:02:35 | 93 | +1 | sí | — | 0 | None | 4 | sin conteo | no registró cruce de línea |
| 09:02:36 | 93 | +1 | sí | — | 0 | None | 48 | sin conteo | no registró cruce de línea |
| 09:02:45 | 93 | +1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 09:02:46 | 93 | +1 | sí | egress(-1) → ingress(+0) → egress(-1) → ingress(+0) → egress(-1) → ingress(+0) → egress(-1) | -1 | egress | 135 | contada |  |

### Ráfaga 5 — 09:29:16–09:29:40 (6 visitas, 6 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:29:16 | 163 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 09:29:21 | 165 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 09:29:27 | 166 | +1 | sí | egress(-1) | -1 | egress | 2 | contada |  |
| 09:29:30 | 168 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 09:29:35 | 169 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 09:29:39 | 170 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |

### Ráfaga 6 — 09:33:02–09:33:37 (10 visitas, 8 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:33:02 | 0 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 09:33:06 | 3 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 09:33:10 | 5 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 09:33:15 | 8 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 09:33:15 | 8 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 09:33:20 | 11 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 09:33:25 | 12 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 09:33:30 | 16 | +1 | sí | egress(-1) | -1 | egress | 2 | contada |  |
| 09:33:36 | 18 | +1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 09:33:37 | 19 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |

### Ráfaga 7 — 09:38:51–09:38:57 (4 visitas, 2 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:38:51 | 28 | +1 | sí | egress(-1) | -1 | egress | 10 | contada |  |
| 09:38:54 | 28 | -1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 09:38:55 | 28 | -1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 09:38:55 | 28 | -1 | sí | ingress(+1) | 1 | ingress | 37 | contada |  |

### Ráfaga 8 — 09:41:52–09:42:13 (4 visitas, 4 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:41:52 | 33 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 09:41:59 | 34 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |
| 09:42:06 | 36 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 09:42:13 | 38 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |

### Ráfaga 9 — 09:57:15–09:57:44 (6 visitas, 6 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 09:57:15 | 70 | +1 | sí | — | -1 | egress | 2 | contada |  |
| 09:57:20 | 72 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 09:57:25 | 73 | +1 | sí | egress(-1) | -1 | egress | 2 | contada |  |
| 09:57:30 | 75 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 09:57:35 | 76 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 09:57:43 | 78 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |

### Ráfaga 10 — 10:01:31–10:01:58 (6 visitas, 6 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 10:01:31 | 2 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 10:01:37 | 3 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 10:01:42 | 6 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 10:01:48 | 7 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 10:01:52 | 9 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 10:01:58 | 10 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |

### Ráfaga 11 — 10:04:55–10:05:18 (4 visitas, 4 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 10:04:55 | 14 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 10:05:03 | 17 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:05:09 | 19 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 10:05:17 | 21 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |

### Ráfaga 12 — 10:09:46–10:10:08 (4 visitas, 4 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 10:09:46 | 27 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 10:09:53 | 29 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:10:01 | 31 | +1 | sí | egress(-1) | -1 | egress | 8 | contada |  |
| 10:10:08 | 33 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |

### Ráfaga 13 — 10:15:46–10:16:14 (11 visitas, 7 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 10:15:46 | 4 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 10:15:47 | 5 | +1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 10:15:53 | 8 | +1 | sí | egress(-1) | -1 | egress | 6 | contada |  |
| 10:15:53 | 9 | +1 | sí | — | 0 | None | 2 | sin conteo | no registró cruce de línea |
| 10:16:00 | 11 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 10:16:00 | 13 | +1 | sí | — | 0 | None | 3 | sin conteo | no registró cruce de línea |
| 10:16:06 | 15 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:16:06 | 14 | +1 | sí | egress(-1) | -1 | egress | 6 | contada |  |
| 10:16:13 | 18 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 10:16:14 | 19 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:16:14 | 18 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |

### Ráfaga 14 — 10:19:32–10:20:02 (9 visitas, 8 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 10:19:32 | 27 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |
| 10:19:38 | 28 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 10:19:39 | 29 | +1 | sí | egress(-1) → ingress(+0) → egress(-1) | -1 | egress | 4 | contada |  |
| 10:19:46 | 31 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 10:19:46 | 32 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:19:53 | 33 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:19:53 | 34 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 10:20:01 | 37 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 10:20:01 | 36 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |

### Ráfaga 15 — 11:30:56–11:31:05 (6 visitas, 2 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 11:30:56 | 134 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 11:31:01 | 135 | +1 | sí | egress(-1) | -1 | egress | 9 | contada |  |
| 11:31:01 | 135 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 11:31:01 | 135 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 11:31:05 | 137 | -1 | sí | — | 0 | None | 3 | sin conteo | no registró cruce de línea |
| 11:31:05 | 137 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |

### Ráfaga 16 — 12:08:12–12:08:17 (4 visitas, 1 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 12:08:12 | 37 | +1 | sí | — | -1 | egress | 3 | contada |  |
| 12:08:13 | 37 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 12:08:17 | 38 | +1 | sí | — | -1 | egress | 1 | suprimida | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 12:08:17 | 38 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |

### Ráfaga 17 — 13:12:56–13:13:12 (8 visitas, 8 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 13:12:56 | 106 | -1 | sí | ingress(+1) | 1 | ingress | 7 | contada |  |
| 13:12:59 | 108 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 13:13:00 | 109 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |
| 13:13:01 | 110 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 13:13:10 | 113 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 13:13:11 | 112 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 13:13:11 | 111 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 13:13:11 | 114 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |

### Ráfaga 18 — 13:54:10–13:54:14 (5 visitas, 5 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 13:54:10 | 172 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 13:54:11 | 174 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |
| 13:54:11 | 175 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 13:54:12 | 176 | -1 | sí | ingress(+1) | 1 | ingress | 6 | contada |  |
| 13:54:14 | 178 | -1 | sí | ingress(+1) | 1 | ingress | 6 | contada |  |

### Ráfaga 19 — 14:52:26–14:52:28 (4 visitas, 3 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 14:52:26 | 279 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 14:52:26 | 279 | -1 | sí | ingress(+1) | 1 | ingress | 7 | contada |  |
| 14:52:26 | 280 | -1 | sí | ingress(+1) | 1 | ingress | 3 | contada |  |
| 14:52:27 | 281 | -1 | sí | ingress(+1) | 1 | ingress | 7 | contada |  |

### Ráfaga 20 — 14:55:08–14:55:15 (4 visitas, 3 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 14:55:08 | 295 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 14:55:11 | 297 | +1 | sí | egress(-1) | -1 | egress | 6 | contada |  |
| 14:55:14 | 298 | -1 | sí | ingress(+1) | 1 | ingress | 2 | contada |  |
| 14:55:15 | 298 | +1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |

### Ráfaga 21 — 15:38:00–15:38:02 (4 visitas, 2 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 15:38:00 | 379 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 15:38:01 | 379 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 15:38:01 | 381 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 15:38:02 | 381 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |

### Ráfaga 22 — 16:11:59–16:12:17 (4 visitas, 4 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 16:11:59 | 421 | +1 | sí | egress(-1) | -1 | egress | 6 | contada |  |
| 16:12:08 | 423 | +1 | sí | egress(-1) | -1 | egress | 7 | contada |  |
| 16:12:10 | 425 | +1 | sí | egress(-1) | -1 | egress | 3 | contada |  |
| 16:12:17 | 427 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |

### Ráfaga 23 — 16:23:20–16:23:34 (4 visitas, 3 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 16:23:20 | 440 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 16:23:29 | 442 | -1 | sí | ingress(+1) | 1 | ingress | 6 | suprimida | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 16:23:33 | 445 | +1 | sí | egress(-1) | -1 | egress | 8 | contada |  |
| 16:23:34 | 447 | +1 | sí | egress(-1) | -1 | egress | 4 | contada |  |

### Ráfaga 24 — 17:21:30–17:21:55 (5 visitas, 5 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 17:21:30 | 514 | -1 | sí | ingress(+1) | 1 | ingress | 4 | contada |  |
| 17:21:40 | 515 | -1 | sí | ingress(+1) | 1 | ingress | 5 | contada |  |
| 17:21:43 | 516 | -1 | sí | ingress(+1) | 1 | ingress | 7 | contada |  |
| 17:21:47 | 519 | -1 | sí | ingress(+1) | 1 | ingress | 6 | contada |  |
| 17:21:55 | 520 | +1 | sí | egress(-1) | -1 | egress | 6 | contada |  |

### Ráfaga 25 — 17:27:11–17:27:18 (4 visitas, 2 con conteo)

| Hora | tid | lado entr. | entr. real | cruces | neto | veredicto | frames reales | resultado | observación |
|---|---:|---:|---|---|---|---|---:|---|---|
| 17:27:11 | 540 | +1 | sí | egress(-1) | -1 | egress | 5 | contada |  |
| 17:27:16 | 542 | -1 | sí | — | 0 | None | 9 | sin conteo | no registró cruce de línea |
| 17:27:17 | 542 | -1 | sí | — | 0 | None | 1 | sin conteo | no registró cruce de línea |
| 17:27:17 | 542 | -1 | sí | ingress(+1) | 1 | ingress | 14 | contada |  |

## Visitas sin conteo

Cada fila es una estadía en la counting zone que no produjo evento. Sin verdad de referencia no puede decidirse si corresponde a un cruce real perdido o a una aproximación que legítimamente no debía contar.

| Hora | tid | entr. → salida | frames reales | motivo |
|---|---:|---|---:|---|
| 08:34:51 | 6 | (593,165) → (653,70) | 5 | no registró cruce de línea |
| 08:35:21 | 11 | (611,313) → (624,81) | 12 | no registró cruce de línea |
| 08:35:22 | 13 | (619,186) → (638,69) | 6 | no registró cruce de línea |
| 08:35:32 | 14 | (613,83) → (609,79) | 32 | balance neto cero (ida y vuelta) |
| 08:35:33 | 15 | (577,288) → (552,322) | 1 | no registró cruce de línea |
| 08:35:39 | 15 | (607,321) → — | — | no registró cruce de línea |
| 08:35:49 | 17 | (594,258) → (577,78) | 1 | no registró cruce de línea |
| 08:54:18 | 62 | (521,157) → (612,52) | 3 | no registró cruce de línea |
| 08:54:46 | 65 | (493,214) → (410,350) | 2 | no registró cruce de línea |
| 09:02:03 | 93 | (737,284) → (911,327) | 5 | no registró cruce de línea |
| 09:02:04 | 93 | (895,313) → (901,322) | 1 | no registró cruce de línea |
| 09:02:05 | 93 | (913,319) → (897,326) | 43 | no registró cruce de línea |
| 09:02:11 | 93 | (891,312) → (897,334) | 10 | no registró cruce de línea |
| 09:02:11 | 93 | (896,317) → (887,324) | 2 | no registró cruce de línea |
| 09:02:12 | 93 | (883,318) → (886,327) | 2 | no registró cruce de línea |
| 09:02:12 | 93 | (893,322) → (886,322) | 4 | no registró cruce de línea |
| 09:02:12 | 93 | (887,321) → (885,322) | 3 | no registró cruce de línea |
| 09:02:14 | 93 | (885,308) → (898,322) | 12 | no registró cruce de línea |
| 09:02:18 | 93 | (885,312) → (889,324) | 6 | no registró cruce de línea |
| 09:02:22 | 93 | (871,315) → (879,322) | 130 | no registró cruce de línea |
| 09:02:33 | 93 | (882,320) → (884,325) | 1 | no registró cruce de línea |
| 09:02:33 | 93 | (906,322) → (903,324) | 3 | no registró cruce de línea |
| 09:02:34 | 93 | (930,318) → (872,338) | 4 | no registró cruce de línea |
| 09:02:35 | 93 | (912,304) → (889,324) | 4 | no registró cruce de línea |
| 09:02:36 | 93 | (909,310) → (885,323) | 48 | no registró cruce de línea |
| 09:02:45 | 93 | (910,321) → (879,323) | 2 | no registró cruce de línea |
| 09:04:01 | 104 | (532,97) → (514,25) | 2 | no registró cruce de línea |
| 09:04:01 | 104 | (503,123) → — | — | no registró cruce de línea |
| 09:06:28 | 107 | (495,141) → (554,59) | 2 | no registró cruce de línea |
| 09:08:53 | 114 | (494,218) → (401,326) | 2 | no registró cruce de línea |
| 09:13:07 | 124 | (680,93) → (665,69) | 1 | no registró cruce de línea |
| 09:13:57 | 127 | (541,88) → (560,64) | 1 | no registró cruce de línea |
| 09:14:00 | 128 | (964,313) → (1022,267) | 4 | no registró cruce de línea |
| 09:17:32 | 140 | (917,300) → (848,363) | 2 | no registró cruce de línea |
| 09:19:48 | 144 | (489,110) → (511,40) | 2 | no registró cruce de línea |
| 09:33:15 | 8 | (790,86) → (815,53) | 1 | no registró cruce de línea |
| 09:33:36 | 18 | (789,241) → (745,375) | 2 | no registró cruce de línea |
| 09:38:54 | 28 | (649,87) → (658,54) | 2 | no registró cruce de línea |
| 09:38:55 | 28 | (670,93) → (674,81) | 2 | no registró cruce de línea |
| 09:40:49 | 32 | (522,117) → (535,70) | 1 | no registró cruce de línea |
| 09:44:12 | 44 | (720,102) → (747,35) | 1 | no registró cruce de línea |
| 10:15:47 | 5 | (488,205) → (487,350) | 2 | no registró cruce de línea |
| 10:15:53 | 9 | (500,223) → (506,329) | 2 | no registró cruce de línea |
| 10:16:00 | 13 | (493,211) → (462,363) | 3 | no registró cruce de línea |
| 10:16:14 | 18 | (456,92) → (467,74) | 1 | no registró cruce de línea |
| 10:19:15 | 23 | (546,105) → (537,60) | 1 | no registró cruce de línea |
| 10:19:46 | 31 | (959,315) → (1012,269) | 1 | no registró cruce de línea |
| 10:20:46 | 38 | (775,83) → (829,47) | 1 | no registró cruce de línea |
| 10:38:23 | 55 | (395,298) → (377,333) | 1 | no registró cruce de línea |
| 10:47:51 | 66 | (544,112) → (577,78) | 1 | no registró cruce de línea |
| 11:03:23 | 92 | (620,85) → (638,63) | 1 | no registró cruce de línea |
| 11:06:10 | 101 | (455,91) → (338,246) | 4 | no registró cruce de línea |
| 11:12:13 | 110 | (666,93) → (683,58) | 1 | no registró cruce de línea |
| 11:22:24 | 119 | (738,102) → (739,69) | 1 | no registró cruce de línea |
| 11:31:01 | 135 | (613,96) → (615,73) | 1 | no registró cruce de línea |
| 11:31:01 | 135 | (619,86) → (622,80) | 1 | no registró cruce de línea |
| 11:31:05 | 137 | (468,99) → (326,243) | 3 | no registró cruce de línea |
| 11:31:05 | 137 | (355,302) → (301,312) | 1 | no registró cruce de línea |
| 11:34:12 | 139 | (782,85) → (792,82) | 3 | no registró cruce de línea |
| 11:34:12 | 139 | (798,84) → (798,71) | 1 | no registró cruce de línea |
| 11:34:37 | 140 | (431,212) → (352,341) | 3 | no registró cruce de línea |
| 11:40:46 | 7 | (564,85) → (574,80) | 1 | no registró cruce de línea |
| 11:42:05 | 9 | (422,313) → (406,348) | 1 | no registró cruce de línea |
| 12:08:13 | 37 | (545,107) → (551,75) | 1 | no registró cruce de línea |
| 12:08:17 | 38 | (859,83) → (858,72) | 1 | no registró cruce de línea |
| 12:14:36 | 50 | (453,149) → (342,274) | 2 | no registró cruce de línea |
| 12:46:25 | 78 | (659,85) → (665,8) | 1 | no registró cruce de línea |
| 13:13:27 | 117 | (614,125) → (630,73) | 1 | no registró cruce de línea |
| 13:13:27 | 116 | (544,101) → (543,72) | 1 | no registró cruce de línea |
| 13:15:49 | 126 | (568,94) → (580,74) | 1 | no registró cruce de línea |
| 13:20:48 | 135 | (651,85) → (658,23) | 1 | no registró cruce de línea |
| 13:55:00 | 181 | (451,155) → (328,245) | 2 | no registró cruce de línea |
| 14:16:00 | 229 | (612,87) → (651,29) | 1 | no registró cruce de línea |
| 14:21:42 | 238 | (688,83) → (685,6) | 1 | no registró cruce de línea |
| 14:24:51 | 243 | (423,100) → (326,244) | 2 | no registró cruce de línea |
| 14:48:13 | 269 | (546,94) → (577,47) | 1 | no registró cruce de línea |
| 14:52:26 | 279 | (500,90) → (471,55) | 1 | no registró cruce de línea |
| 14:53:52 | 290 | (491,321) → (669,79) | 2 | no registró cruce de línea |
| 14:55:15 | 298 | (374,320) → (360,347) | 1 | no registró cruce de línea |
| 15:00:20 | 314 | (992,84) → (1019,219) | 3 | no registró cruce de línea |
| 15:00:46 | 317 | (936,87) → (1008,240) | 4 | no registró cruce de línea |
| 15:01:48 | 324 | (968,184) → (1009,110) | 4 | no registró cruce de línea |
| 15:04:52 | 331 | (963,94) → (1008,176) | 6 | no registró cruce de línea |
| 15:07:52 | 348 | (999,196) → (950,63) | 8 | no registró cruce de línea |
| 15:38:01 | 379 | (858,98) → (883,74) | 1 | no registró cruce de línea |
| 15:38:02 | 381 | (540,91) → (554,75) | 1 | no registró cruce de línea |
| 16:00:25 | 407 | (993,101) → (977,71) | 1 | no registró cruce de línea |
| 16:12:31 | 428 | (611,84) → (630,55) | 1 | no registró cruce de línea |
| 16:41:23 | 461 | (501,206) → (331,356) | 2 | no registró cruce de línea |
| 16:53:17 | 472 | (490,184) → (533,70) | 3 | no registró cruce de línea |
| 17:19:11 | 505 | (429,235) → (386,332) | 1 | no registró cruce de línea |
| 17:27:16 | 542 | (544,93) → (556,42) | 9 | no registró cruce de línea |
| 17:27:17 | 542 | (588,95) → (600,65) | 1 | no registró cruce de línea |
| 17:28:48 | 549 | (617,82) → (631,62) | 1 | no registró cruce de línea |
| 17:29:17 | 550 | (559,83) → (576,20) | 1 | no registró cruce de línea |
| 17:36:26 | 594 | (1003,169) → (1043,93) | 1 | no registró cruce de línea |
| 18:01:23 | 632 | (640,309) → (614,74) | 2 | no registró cruce de línea |
| 18:01:24 | 633 | (639,291) → (613,324) | 2 | no registró cruce de línea |
| 18:16:33 | 650 | (480,96) → (430,322) | 1 | no registró cruce de línea |
| 18:47:30 | 664 | (537,296) → (526,329) | 1 | no registró cruce de línea |
| 18:52:44 | 668 | (623,83) → (623,78) | 1 | no registró cruce de línea |
| 18:53:34 | 672 | (609,97) → (606,79) | 2 | no registró cruce de línea |
| 08:33:20 | 4 | (555,129) → (343,299) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 08:34:56 | 7 | (569,101) → (567,333) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 08:40:07 | 23 | (458,221) → (496,31) | 2 | exit_kalman_skipped (reason=no_outside_history net=[-1]) |
| 08:52:14 | 56 | (538,248) → (799,51) | 1 | exit_kalman_skipped (reason=no_outside_history net=[-1]) |
| 08:52:52 | 58 | (495,99) → (419,337) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 08:52:56 | 61 | (488,101) → (420,351) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 08:54:46 | 64 | (526,174) → (426,335) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[1]) |
| 09:03:16 | 98 | (863,157) → (759,326) | 3 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 09:21:25 | 154 | (526,88) → (426,333) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 09:21:59 | 155 | (540,93) → (418,345) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 09:50:21 | 54 | (612,126) → (494,367) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 10:01:15 | 0 | (512,299) → (543,72) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 10:50:35 | 72 | (593,123) → (371,364) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 10:50:51 | 73 | (570,108) → (408,347) | 6 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 10:51:02 | 76 | (453,90) → (404,358) | 6 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 11:08:52 | 104 | (533,92) → (413,337) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 11:11:16 | 107 | (509,135) → (349,338) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 11:51:37 | 20 | (534,94) → (363,331) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 12:06:29 | 33 | (536,120) → (372,374) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 12:08:17 | 38 | (816,320) → (841,35) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 13:13:26 | 116 | (398,297) → (506,36) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 13:20:47 | 135 | (655,188) → (681,79) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 13:47:24 | 152 | (856,154) → (852,343) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[1]) |
| 14:00:12 | 199 | (485,89) → (393,333) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 14:16:44 | 232 | (531,99) → (343,368) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 14:18:38 | 235 | (483,98) → (314,341) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 14:32:22 | 247 | (552,128) → (612,67) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 15:34:18 | 373 | (461,246) → (590,41) | 4 | exit_kalman_skipped (reason=no_outside_history net=[-1]) |
| 16:00:26 | 408 | (438,84) → (341,342) | 5 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 16:23:29 | 442 | (547,85) → (427,336) | 6 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 16:30:02 | 453 | (462,91) → (384,356) | 6 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 17:18:06 | 501 | (570,88) → (451,341) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 17:20:21 | 509 | (547,126) → (519,330) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 17:24:35 | 527 | (510,131) → (547,39) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 18:13:00 | 643 | (676,294) → (888,67) | 1 | exit_thin_evidence_skipped (real_inside_frames=1 threshold=2 net=[-1]) |
| 18:42:13 | 656 | (769,112) → (817,374) | 4 | exit_kalman_skipped (reason=no_outside_history net=[1]) |
| 18:52:42 | 668 | (610,298) → (623,80) | 35 | exit_short_height_skipped (height_m=0.96 threshold=1.00 net=[-1]) |
| 18:53:07 | 669 | (623,309) → (623,77) | 16 | exit_short_height_skipped (height_m=0.96 threshold=1.00 net=[-1]) |
| 18:53:57 | 673 | (589,293) → (599,80) | 25 | exit_short_height_skipped (height_m=0.97 threshold=1.00 net=[-1]) |

CSV escrito en <ruta-local>
