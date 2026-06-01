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
`death_emit_count`, etc.). `track_stitching_ratio` y `death_emit_count` se
resetean a diario en el device (rollover de medianoche en `main.py`, que llama
`counter.reset_daily()`); `ghost_adoption_count` es **acumulativo desde el boot
del proceso** (el tracker no tiene `reset_daily`) — leelo como contador
monotónico, no como total del día.

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

### 5. FP no-humano cuenta (perro, carrito, objeto sobre silla)

**Lectura**: el detector YOLO está disparando sobre algo que no es persona —
perro caminando, carrito alto, mochila/campera sobre una silla. Si el objeto
cruza la línea (perro caminando), el counter dispara un IN/OUT espurio.

**Síntoma diagnóstico**: counts en horarios sin clientes, o counts
inconsistentes con la observación humana del operador. Buscar en logs:

```bash
# Eventos de conteo con altura muy baja (height_m < 1.0). En el JSON del log
# height_m es float; este grep agarra los que arrancan con "0." o "0.0..."
sudo journalctl -u people-counter --since "1 hour ago" \
    | grep '"counting_event_published"' | grep -E '"height_m": 0\.[0-9]'
```

O contra RDS — la función SQL `height_class()` clasifica los `height_m`
crudos como `'child'` (< 1.55m), filtrá los muy bajos:

```sql
SELECT bucket_15min, COUNT(*) AS n_short
FROM count_events
WHERE store_id = '<store-id>'
  AND bucket_day = CURRENT_DATE
  AND height_m < 1.0
GROUP BY bucket_15min ORDER BY bucket_15min DESC;
```

Si la mayoría de los counts del día tienen `height_m < 1.0` → casi seguro
son FPs no-humanos.

**Fix**:

```yaml
counter:
  min_count_height_m: 1.0   # de 0 → activa el filtro
```

Con 1.0, el counter rechaza emit cuando la mediana de altura del track es
< 1 m AND la medición existe. Tracks con altura desconocida (SGBM falló por
motion blur / oclusión) PASAN — preferimos recall sobre precisión en
ambigüedad. El preview también oculta tracks debajo del threshold (refleja
lo que el pipeline procesa).

**Trade-off**: niños muy chicos (< 4 años, altura ~0.95 m) no se cuentan.
En retail típico estos vienen con padre (que sí se cuenta), así que el
trade-off es aceptable.

**Knobs alternativos**:
- `0.7` más conservador (filtra solo perros < 70 cm).
- `1.2` filtra hasta niños de ~6 años (riesgo de perder counts legítimos).

Confirmar en logs que el filtro está agarrando lo esperado:

```bash
sudo journalctl -u people-counter --since "10 min ago" \
    | grep -E "short_height_skipped"
```

#### 5b. El FP no-humano NO tiene altura (perro que SGBM no mide)

`min_count_height_m` solo filtra cuando la altura **existe** y es baja — tracks
con altura desconocida (mediana = `None`) PASAN, para preservar recall cuando
SGBM falla en una persona real (motion blur / oclusión). Pero muchos FPs
no-humanos salen JUSTAMENTE sin altura: SGBM no le saca cabeza confiable a un
perro (forma/altura atípica) ni a un track fantasma sobre clutter. Esos se
cuelan por el agujero del guard de altura.

**Síntoma diagnóstico**: counts en horarios sin clientes, todos con
`height_m=?` (sin altura) y confianza baja. Caso real piloto (2026-06-01): de
lunes a jueves 10:30-18:30 el único "tráfico" es un perro. Análisis de 831
`count_events`: **todo** evento con altura medida tiene `conf >= 0.5` y es
humano; el perro/fantasma sale **siempre sin altura y con `conf <= 0.56`**.

```sql
-- Crosstab confianza × ¿SGBM midió altura? — la firma del FP salta a la vista.
SELECT width_bucket(confidence, 0, 1, 10) * 0.1 AS conf_bucket,
       count(*) FILTER (WHERE height_m IS NULL)     AS sin_altura,
       count(*) FILTER (WHERE height_m IS NOT NULL) AS con_altura
FROM count_events WHERE store_id = '<store-id>'
GROUP BY 1 ORDER BY 1;
```

Si todo lo de `conf < ~0.55` cae en `sin_altura` → es el patrón del perro/fantasma.

**Fix** (complementa, no reemplaza, a `min_count_height_m`):

```yaml
counter:
  min_count_confidence: 0.60   # default; sin altura + conf < esto → no cuenta
```

Sin altura medida Y `mediana(conf) < 0.60` → el counter rechaza el emit. Una
persona real sin altura trae conf alta (>= 0.6) y **pasa** → recall preservado.
Es un eje **ortogonal** a `height_confidence_gate` (ese solo decide si reportar
demografía cuando HAY altura; el conteo se mantiene). Acá, sin altura, decide
contar-o-no.

**Knobs**:
- `0.50` prioriza recall (deja pasar FPs de conf 0.50-0.60, ej. el perro tope ~0.56).
- `0.0` desactiva el guard (back-compat: cualquier track sin altura cuenta).

```bash
# Agarra las dos ramas: "exit_lowconf_noheight_skipped" y "reason=lowconf_noheight".
sudo journalctl -u people-counter --since "10 min ago" \
    | grep -E "lowconf_noheight"
```

### 6. Clutter estructural intenso (tiendas de ropa, sites con perchero/mostrador)

**Lectura**: el frame tiene zonas con detecciones espurias persistentes —
percheros con ropa que la gente mueve, mostradores con maniquíes, vidrieras
con tráfico exterior visible. Los guards anti-FP downstream (height,
real_inside_frames) atrapan algo pero el preview sigue ruidoso y el costo
Hungarian del tracker se infla con tracks que no van a contar nunca.

**Fix arquitectural**: activar el filtro pre-tracker por polígono. Descarta
detecciones fuera de la "tracking_zone" (más amplia que el counting_zone,
con margen de approach) ANTES de que entren al tracker.

**Tres modos de definición** (mutuamente exclusivos, evaluados en este
orden de precedencia: `polygon > frame_margin_px > auto_margin_px`):

**Modo recomendado: `frame_margin_px`** (predecible, simétrico):

```yaml
tracking:
  tracking_zone:
    enabled: true
    frame_margin_px: 100   # excluye 100 px desde cada borde del frame
```

Funciona independientemente del tamaño del counting_zone. Si el margen
chocaría con el counting_zone, se reduce automáticamente preservando un
buffer de lead-in de 30 px. Recomendado para "modo conservador por
precaución" — filtra cualquier clutter periférico (perchero, mostrador
lateral, banner publicitario arriba, vidriera) en un solo knob.

**Modo `auto_margin_px`** (expand desde counting_zone):

```yaml
tracking:
  tracking_zone:
    enabled: true
    auto_margin_px: 250    # ~1 m de lead-in del approach
```

Útil cuando el counting_zone es chico y centrado. **Caveat**: si el
counting_zone es grande relativo al frame, el margen choca con los bordes
y el polígono resultante es el frame entero (sin efecto). En ese caso
usar `frame_margin_px`.

**Modo `polygon` manual** (geometría arbitraria, control fino):

```yaml
tracking:
  tracking_zone:
    enabled: true
    polygon:
      - [200, 80]    # esquina superior izquierda
      - [1050, 80]
      - [1050, 580]
      - [200, 580]   # esquina inferior izquierda
```

Para sites con geometría compleja (entrada diagonal, exclusión específica
de una vidriera). Tiene la máxima precedencia.

**Verificar en logs que se aplicó:**

```bash
sudo journalctl -u people-counter --since "1 min ago" \
    | grep -E "tracking_zone_polygon"
```

Buscás uno de estos:
- `tracking_zone_polygon_from_frame_margin margin=Xpx ...`
- `tracking_zone_polygon_from_auto_margin margin=Xpx ...`
- (polygon manual no loguea — verificar config)

**Trade-off**: si el polígono resultante apenas alcanza el counting_zone,
una persona caminando rápido que aparezca directamente en el counting_zone
sin frame previo en la tracking_zone podría no tener `last_outside_pos`
válido. Subir el margen (más amplio = más lead-in).

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
| `min_count_height_m` | 0.0 (off) | `counter` | FPs no-humanos cuentan | filtra niños chicos legítimos |
| `min_count_confidence` | 0.60 | `counter` | FPs sin altura (perro/fantasma) cuentan | personas sin altura no cuentan (subir recall) |
| `min_real_inside_frames` | 0 (off) | `counter` | FPs single-frame al borde del counting zone | caminantes muy rápidos (<150 ms inside) se pierden |
| `tracking_zone.enabled` | false (off) | `tracking` | clutter estructural genera tracks ruidosos | counting_zone + margen pequeños podrían perder approach |
| `tracking_zone.frame_margin_px` | 100 | `tracking` | filtrar más clutter periférico (subir) | menor margen es más permisivo (bajar si pierde approach) |
| `tracking_zone.auto_margin_px` | 250 | `tracking` | filtrar más agresivo (mayor margen pierde más) | menor margen pierde más approach lead-in |
| `height_confidence_gate` | 0.5 | `counter` | demografía espuria en eventos | demografía de pasadas rápidas se reporta unknown |
| `keepalive_max_frames` | 600 | `tracking.state_machine` | rara vez | counts fantasma |
| `ambiguous_match_ratio` | 0.8 | `tracking.state_machine` | rara vez | ID swaps |

Para la filosofía del rescue cascade (las 3 capas + entry-Kalman guard),
ver la sección **Design philosophy del counter** en `CLAUDE.md`. Para la
matriz de cobertura de tests (qué celda del producto cartesiano cubre cada
test), ver [`docs/counter_test_matrix.md`](counter_test_matrix.md).
