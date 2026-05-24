# Tracker tuning — runbook operacional

Guía reactiva para tunear el tracker y el counter cuando la telemetría del
piloto / la flota muestra patrones que sugieren un knob mal calibrado para
un site específico. Defaults shipeados son baseline conservadora — esta guía
explica **cómo detectar** que un site necesita override y **qué cambiar**.

> **Quién la usa**: el operador del piloto o quien hace SRE de la flota.
> **Cuándo**: cuando la telemetría dispara una alerta o el cliente reporta
> "no contó a un visitante" / "contó de más".
> **Cómo aplicar un fix**: editar `/etc/people-counter/config.yaml` en el
> device y `sudo systemctl restart people-counter`. **No** reflashear, no
> tocar código.

---

## Canaries de observabilidad

El device publica cada 5 minutos en `telemetry`:

| Métrica | Significado | Ideal | Umbral de alerta |
|---|---|---|---|
| `track_stitching_ratio` | `unique_track_ids / total_counts` del día | ≈ 1.0 | > 1.3 sostenido 1 h |
| `death_emit_count` | Death-emits disparados hoy (capa 3 del rescue) | bajo | depende del ratio |
| `ghost_adoption_count` | Adopciones de ID exitosas (capa 1, acumulativo) | depende | — |

Las tres se persisten en RDS (`telemetry.track_stitching_ratio`,
`death_emit_count`, etc.). Reset diario en el device.

**Lookup table del árbol diagnóstico**:

| `stitching_ratio` | `adoption` | `death_emit` | Diagnóstico |
|---|---|---|---|
| ≈ 1.0 | 0 | 0 | Tracker perfecto |
| ≈ 1.0 | > 0 | 0 | Fragmentación silenciosa rescatada por capa 1 |
| ≈ 1.0 | 0 | > 0 | Crossers rescatados por capa 3 (death-emit) |
| > 1.3 | 0 | 0 | 🚨 FRAGMENTACIÓN SIN RESCATE (alarma) |
| > 1.3 | > 0 | > 0 | Tracker flakey, ambas capas activas (OK pero detector flojo) |

---

## Cómo consultar los canaries

### En cloud (Grafana / RDS)

```sql
-- Ratio + counts por device, últimas 24 h
SELECT device_id,
       MAX(track_stitching_ratio)     AS ratio_max,
       MAX(death_emit_count)          AS death_emit,
       MAX(ghost_adoption_count)      AS adoption
FROM   telemetry
WHERE  ts > NOW() - INTERVAL '24 hours'
GROUP  BY device_id
ORDER  BY ratio_max DESC NULLS LAST;
```

### En el device (SSH)

```bash
# Logs en vivo del tracker (TRACKDBG = eventos diagnósticos)
sudo journalctl -u people-counter -f | grep TRACKDBG

# Histórico last 1h, agrupado por tipo de evento
sudo journalctl -u people-counter --since "1 hour ago" \
    | grep TRACKDBG \
    | sed 's/.*TRACKDBG //' | awk '{print $1}' | sort | uniq -c | sort -rn
```

Eventos TRACKDBG típicos:

| Evento | Significado |
|---|---|
| `entry` | Track entró fresco a la counting zone (snapshot de sides) |
| `entry_kalman_skipped` | Primer frame inside fue pura predicción Kalman; entry diferida |
| `cross` | Cruce de línea registrado in-zone con detección real |
| `exit` | Track salió de la counting zone con verdict de signo del net |
| `exit_kalman_skipped` | Exit por Kalman rechazado por guard `had_outside_pos` |
| `death_emit_skipped` | Death-emit rechazado por guard. `reason=no_outside_history` o `small_visit_range` |
| `ghost_adopted` | Capa 1 del rescue funcionó: track nuevo heredó ID + meta del ghost |
| `ghost_outside_invalidated` | Ghost adoption descartó un `last_outside_pos` heredado por estar lejos |

---

## Patrones síntoma → diagnóstico → fix

Los 4 patrones más comunes. Cada uno: cómo detectarlo, qué evento TRACKDBG
lo confirma, qué knob ajustar.

### 1. Fragmentación sin rescate (`stitching_ratio > 1.3` + `adoption ≈ 0`)

**Lectura**: el detector pierde y re-detecta, spawneando IDs nuevos sin que
la capa 1 logre adoptarlos. Los counts del día son inflados (1 persona
real = 2-3 IDs distintos contados separadamente, o crossings perdidos).

**Confirmar en logs**: pocos `ghost_adopted`, muchos tracks nuevos
spawneando sin ghosts disponibles, o ghosts adoptables que no pasan el
gate de IoU/distancia.

**Fix**:

```yaml
tracking:
  state_machine:
    adoption_window_frames: 60    # de 30 → ~3 s @ 20 fps
    # mantener adoption_iou_min y adoption_max_dist_px en defaults
```

Si tras subir la ventana sigue ≈ 0 adopciones, relajar el gate:

```yaml
    adoption_iou_min: 0.2         # de 0.3 → más permisivo
```

⚠ **No bajar `adoption_iou_min` < 0.2** sin verificar primero que no hay
ID swaps visibles en cruces. Es el gate que nos diferencia del incumbent
FFC (que usa booleano puro y sí tiene swaps).

### 2. Crossers perdidos en la zona de la línea (no contó al visitante)

**Lectura**: el detector dropea a la persona justo cuando está cruzando.
Las capas 2 y 3 del rescue (decisive Kalman cross + death-emit) deberían
agarrarlo, pero el guard de `visit_range >= 80` lo rechaza porque la
persona tuvo muy poca observación real.

**Síntoma diagnóstico exacto**: `death_emit_count = 0` con
`track_stitching_ratio > 1.3` sostenido. Confirmar con logs:

```bash
sudo journalctl -u people-counter --since "1 hour ago" \
    | grep "death_emit_skipped" | grep "small_visit_range" | wc -l
```

Si el contador es alto → el guard 2 del death-emit está rechazando
crossers reales con poca observación.

**Fix**:

```yaml
counter:
  min_visit_range_for_death_emit: 50.0   # de 80 → más permisivo
```

⚠ **No bajar < 40** sin observar el otro lado: el guard existe para no
contar sitters cuyo bbox jitterea sobre la línea (cabeza moviéndose
~20-30 px). Bajar mucho introduce FPs de sitter.

### 3. Counts fantasma (contó de más)

**Lectura**: el counter emitió un evento sin pasada real. Generalmente
porque un guard del rescue cascade no agarró un caso edge.

**Confirmar en logs**: buscar `cross` o `exit` con `verdict ingress/egress`
asociados a `track_id` que NO tienen `entry` real previo (entry hecha por
Kalman) o tracks que nacen dentro de la counting zone y emiten.

```bash
# Tracks que emitieron con muy poca historia outside
sudo journalctl -u people-counter --since "1 hour ago" \
    | grep "exit" | grep "verdict=" | head -20
```

**Fix**: subir el cap del rescue (más conservador):

```yaml
counter:
  min_visit_range_for_death_emit: 100.0  # de 80 → más estricto
```

Si persiste, considerar reducir keep-alive en la counting zone (el track
PENDING sobrevive demasiado y extrapola un cruce espurio):

```yaml
tracking:
  state_machine:
    keepalive_max_frames: 300              # de 600 (~24 s) → ~12 s
```

### 4. ID swaps en cruces densos

**Lectura**: dos personas adyacentes con tracks que se intercambian IDs
durante un cruce. La capa 1 (ghost adoption) puede estar adoptando con un
candidato equivocado.

**Confirmar**: revisar el viewer durante el cruce; los IDs deberían
mantenerse estables. En telemetría es invisible (no hay canary específico
para swaps), por eso requiere observación humana.

**Fix**:

```yaml
tracking:
  state_machine:
    adoption_iou_min: 0.5                  # de 0.3 → más exigente
    ambiguous_match_ratio: 0.7             # de 0.8 → rechaza matches
                                           # más ambiguos en cruces
```

---

## Notas operacionales

- **Aplicar un fix por vez** y observar 24 h antes de combinar cambios.
  Esto vale para todo este runbook — cambiar dos knobs simultáneos en un
  site oculta cuál fue el efectivo.
- **Documentar el override per-site**. El config canónico shipeado tiene
  defaults; cada device en producción puede divergir. Mantener un registro
  (ej. spreadsheet) de qué site usa qué override y por qué.
- **Re-evaluar al revertir**. Si el "fix" deja de hacer falta (cambió el
  layout del local, se mejoró el detector con un re-train), volver al
  default. Defaults sostenibles > overrides perpetuos.
- **Si nada de esto encaja**, mirar `docs/database_schema.md` para queries
  más finas sobre `count_events` (per-track histogramas de
  `visit_range_max`, etc.) y abrir un issue con los logs TRACKDBG del
  evento problemático.

---

## Referencia de knobs (resumen)

| Knob | Default | Sección config | Cuándo subir | Cuándo bajar |
|---|---|---|---|---|
| `adoption_window_frames` | 30 | `tracking.state_machine` | `stitching_ratio > 1.3` | rara vez |
| `adoption_iou_min` | 0.3 | `tracking.state_machine` | ID swaps visibles | falta de rescue |
| `adoption_max_dist_px` | 100 | `tracking.state_machine` | low FPS site | nunca |
| `ghost_outside_invalidate_px` | 150 | `tracking.state_machine` | rara vez | rara vez |
| `min_visit_range_for_death_emit` | 80 | `counter` | counts fantasma | crossers perdidos |
| `keepalive_max_frames` | 600 | `tracking.state_machine` | rara vez | counts fantasma |
| `ambiguous_match_ratio` | 0.8 | `tracking.state_machine` | rara vez | ID swaps |

Para la filosofía del rescue cascade (las 3 capas + entry-Kalman guard),
ver la sección **Design philosophy del counter** en `CLAUDE.md`. Para la
matriz de cobertura de tests (qué celda del producto cartesiano cubre cada
test), ver [`docs/counter_test_matrix.md`](counter_test_matrix.md).
