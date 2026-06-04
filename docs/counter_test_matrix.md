# Counter — matriz de cobertura discriminante

Mapeo entre las **dimensiones que el código del counter discrimina** (los
`if/else` reales de `_process_track`, `_emit_on_death`, `_decisive_kalman_cross`,
`check_all`) y los tests que cubren cada celda significativa del producto
cartesiano. Sirve para:

- **Auditoría de cobertura**: identificar gaps reales del test suite sin
  inflar con tests redundantes.
- **Onboarding**: el próximo dev ve el shape del comportamiento + cómo lo
  ejercita la suite.
- **Trazabilidad regulatoria**: "demostrá que el counter es correcto" se
  responde con la matriz + ~920 tests verde + canaries en flota.

> **NO es exhaustivo combinatorio.** El producto teórico es ~86.000 celdas;
> la mayoría son trivially equivalent (simetrías horizontal/vertical,
> ingress/egress) o structurally void (CANDIDATE nunca cuenta). El matrix
> enumera ~70 celdas que el código **bifurca de forma única** y que un test
> dedicado tiene valor distinto a los demás.

---

## Dimensiones discriminantes

| Eje | Valores | Donde se bifurca en el código |
|---|---|---|
| **A — Entry source** | A1 real / A2 Kalman-skip / A3 born-inside | `_process_track` líneas 866-922 |
| **B — Exit type** | B1 real / B2 Kalman / B3 death-with-cross / B4 death-no-cross / B5 resurrected mid-life | `_process_track` ramas exit; `_emit_on_death`; `check_all` resurrection |
| **C — Balance neto del cruce** | C1 +1 / C2 -1 / C3 0 con zigzag / C4 0 sin cruces / C5 \|net\| ≥ 2 | `_emit_on_death` short-circuit `not any(n != 0)`; `_process_track` verdict por signo |
| **D — `had_outside_pos`** | D1 True / D2 False | Guards en `_emit_on_death` y exit-Kalman; condicional en entry-fresca |
| **E — `visit_range` vs MIN** | E1 ≥ MIN (80 default) / E2 < MIN | Guard 2 de `_emit_on_death` |
| **F — Decisive Kalman cross** | F1 decisive (`disappeared ≤ 15` AND `displacement ≥ 30`) / F2 not-decisive | `_decisive_kalman_cross` |
| **G — Cantidad de líneas** | G1 1 / G2 N (cada una con net independiente) | `_process_track` loop sobre `self._lines` |
| **H — Labels** | H1 two-way / H2 one-way (sticky) | `Line.crossing_label` retorna `None` para dirección sin label |
| **I — Orientación** | I1 horizontal / I2 vertical | `Line.__post_init__`, `side_of`, `within_segment` |
| **J — Track state** | J1 CONFIRMED / J2 PENDING / J3 CANDIDATE (rechazo) | Primera línea de `_process_track` |
| **K — Identity** | K1 same-id / K2 ghost-adopted (capa 1 rescue) / K3 new-id | `EuclideanTracker._try_adopt_ghost` (decisión IoU/dist) + `_resurrect_ghost` (aplica meta) |
| **L — Keep-alive** | L1 active (track PENDING extrapolando inside) / L2 not | `_inside_keepalive_counting_zone` + guard de promoción LOST en `_record_miss` |
| **M — Ghost outside_pos inherit** | M1 preserved (dist ≤ threshold) / M2 invalidated (dist > threshold) | `_resurrect_ghost` con `ghost_outside_invalidate_px` |
| **N — Gate de altura humana** | N1 altura ≥ `min_count_height_m` (o gate off) / N2 altura < umbral (rechaza) / N3 altura = None (pasa, no aplica) | Guard `min_count_height_m` en exit (`_process_track`) y death (`_emit_on_death`) |
| **O — Gate conf sin altura** | O1 conf ≥ `min_count_confidence` (o gate off) / O2 conf < umbral con altura None (rechaza) / O3 altura presente (gate ignorado) | Guard `min_count_confidence` en exit y death |
| **P — Gate demografía** | P1 conf ≥ `height_confidence_gate` (reporta demografía) / P2 conf < umbral (height_m/head_depth_m → None, demografía unknown; **NO afecta conteo**) | `height_confidence_gate` en emisión del payload |
| **Q — Evidencia real inside** | Q1 `real_inside_frames` ≥ `min_real_inside_frames` (o gate off) / Q2 < umbral (rechaza, anti-flicker single-frame) | Guard `min_real_inside_frames` en exit y death |

Ejes ortogonales del counter en sí: **A × B × C × D × E × F × G × H × I × J**.
Ejes que cruzan al tracker pero afectan el counter: **K × L × M**.
Ejes de los gates anti-FP (filtros de emisión, independientes entre sí): **N × O × P × Q**.

---

## Matriz reducida — celdas significativas

Notación: `✓` cubierto / `gap` requiere test / `void` structurally impossible /
`mirror` cubierto por simetría.

### Cuadrante B1 — exit real observado (camino feliz)

| # | A | B | C | D | E | F | G | Resultado | Test |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A1 | B1 | C1 | D1 | E1 | - | G1 | emit ingress | `test_ingress_counts_on_exit` |
| 2 | A1 | B1 | C2 | D1 | E1 | - | G1 | emit egress | `test_egress_counts_on_exit` |
| 3 | A1 | B1 | C3 | D1 | E1 | - | G1 | no emit (zigzag cancela) | `test_indeciso_with_two_way_line_cancels_net_zero`, `test_oscillation_nets_to_zero_without_debounce` |
| 4 | A1 | B1 | C4 | D1 | E1 | - | G1 | no emit (no cruce) | `test_indeciso_no_crossing`, `test_oscillation_without_full_cycle_does_not_count_twice` |
| 5 | A1 | B1 | C5 | D1 | E1 | - | G1 | emit por signo del net | `test_real_cycle_with_double_crossing_same_direction_counts_once` **(gap-fill)** |
| 6 | A3 | B1 | C1 | D2 | E1 | - | G1 | emit ingress (born inside cruza y sale) | `test_entry_side_fallback_when_track_born_inside_counting_zone` |
| 7 | A1 | B1 | C1 | D1 | E1 | - | G1 | emit con entry desde approach lateral | `test_lateral_entry_crosses_line_lateral_exit_counts`, `test_lateral_entry_crosses_line_downside_egress` |
| 8 | A1 | B1 | C1 | D1 | E1 | - | G2 | emit (multi-line, una con cruce) | `test_two_one_way_lines_separate_doors`, `test_multi_line_each_line_tracks_independent_net` **(gap-fill)** |
| 9 | A1 | B1 | C1 | D1 | E1 | - | G1 + I2 | emit (vertical line) | `test_vertical_line_ingress`, `test_vertical_line_egress` |
| 10 | - | - | - | - | - | - | G1 + sin counting_zone | no emit | `test_no_counting_zone_single_crossing_does_not_count` |

### Cuadrante B2 — exit por Kalman extrapolation

| # | A | B | C | D | E | F | Resultado | Test |
|---|---|---|---|---|---|---|---|---|
| 11 | A1 | B2 | C1 | D1 | E1 | F1 | emit (decisive) | `test_decisive_kalman_cross_at_exit_counts`, `test_kalman_exit_counts_when_track_has_outside_history`, `test_crossing_real_then_exit_on_prediction_still_counts` |
| 12 | A1 | B2 | C1 | D1 | E1 | F2 | no emit (drift) | `test_kalman_drift_at_exit_does_not_count` |
| 13 | A1 | B2 | C1 | D1 | E1 | F1 | no emit (disappeared > 15) | `test_kalman_cross_too_old_does_not_count` |
| 14 | A3 | B2 | C1 | D2 | E1 | F1 | no emit (born inside, exit Kalman) | `test_kalman_exit_skipped_when_track_born_inside_counting_zone` |
| 15 | A1 | B2 | C1 | D2 | E1 | F1 | no emit (no outside hist) | implícito en #14 (D2 = had_outside=False) |
| 16 | A1 | B2 | C1 | D1 | E1 | F1 | emit aunque exit por Kalman si hay real cross previo | `test_real_detection_cross_at_exit_still_counts` |

### Cuadrante B3 — death-in-zone con cruce registrado

| # | A | B | C | D | E | Resultado | Test |
|---|---|---|---|---|---|---|---|
| 17 | A1 | B3 | C1 | D1 | E1 | emit post grace | `test_count_on_track_death_inside_counting_zone_after_crossing`, `test_death_inside_counting_zone_after_crossing_emits_count`, `test_death_emit_with_outside_history`, `test_death_emit_fires_after_grace_expires` |
| 18 | A1 | B3 | C1 | D2 | E1 | no emit (guard 1) | `test_death_emit_skipped_no_outside`, `test_death_emit_skipped_when_track_spawned_inside_counting_zone` |
| 19 | A1 | B3 | C1 | D1 | E2 | no emit (guard 2) | `test_death_emit_skipped_small_visit_range`, `test_death_emit_skipped_when_visit_range_too_small` |
| 20 | A1 | B3 | C3 | D1 | E1 | no emit (net zigzag = 0) | `test_death_with_zigzag_net_zero_does_not_emit` **(gap-fill)** |
| 21 | A3 | B3 | C1 | D2 | E1 | no emit (born inside) | `test_death_emit_skipped_when_track_spawned_inside_counting_zone` |
| 22 | A1 | B3 | C5 | D1 | E1 | emit por signo | implícito (mismo path que C1; gap-fill #5 valida invariante del signo) |
| 23 | A1 | B3 | C1 | D1 | E1 | knob override | `test_min_visit_range_override_relaxes_guard_2`, `test_build_counter_reads_min_visit_range_from_config` |

### Cuadrante B4 — death-in-zone sin cruce

| # | A | B | C | D | E | Resultado | Test |
|---|---|---|---|---|---|---|---|
| 24 | A1 | B4 | C4 | * | * | no emit (early return en `_emit_on_death`) | `test_lost_inside_without_crossing_does_not_count`, `test_death_inside_counting_zone_without_crossing_no_count` |

### Cuadrante B5 — track resucitado mid-life

| # | A | B | C | D | E | Resultado | Test |
|---|---|---|---|---|---|---|---|
| 25 | A1 | B5 | C1 | D1 | E1 | death-emit cancelado, emit natural al exit | `test_death_emit_deferred_then_resurrected_no_double_count` |
| 26 | K2 | - | - | - | - | meta heredado habilita emit por adoption | `test_ghost_adoption_preserves_counter_meta_so_resurrected_track_emits` **(gap-fill)** |
| 27 | A1 | B1 | C1 | D1 | E1 | exit limpio + muerte posterior no double-count | `test_clean_exit_then_death_does_not_double_count` |

### Cuadrante A2 — entry Kalman skipped

| # | A | B | C | D | E | F | Resultado | Test |
|---|---|---|---|---|---|---|---|---|
| 28 | A2 | * | * | * | * | * | entry diferida, no dispara nada hasta real | `test_entry_fresca_skipped_when_first_inside_frame_is_kalman` |
| 29 | A2 → A1 | B1 | C1 | D1 | E1 | - | emit normal cuando llega real | `test_entry_fresca_deferred_until_real_detection` |
| 30 | - | - | - | - | - | - | `last_outside_pos` solo con detecciones reales | `test_last_outside_pos_only_updated_with_real_detections` |

### Estados del track + line semantics

| # | Caso | Test |
|---|---|---|
| 31 | J2 PENDING cuenta | `test_pending_state_counted` |
| 32 | J3 CANDIDATE rechazado | `test_candidate_not_counted` |
| 33 | H2 one-way + cross sin label | `test_indeciso_with_one_way_line_does_not_count` |
| 34 | Cross fuera del segment | `test_crossing_outside_segment_is_ignored` |
| 35 | Frame de Kalman puro inside no registra cross | `test_inside_was_inside_prediction_does_not_register_cross` |
| 36 | Two full cycles del mismo track | `test_same_track_two_full_cycles_both_count` |
| 37 | IN seguido de OUT del mismo track | `test_in_followed_by_out_both_count` |
| 38 | Debounce filtra jitter | `test_debounce_filters_jitter_around_line` |
| 39 | Debounce no bloquea cross real | `test_debounce_does_not_block_real_crossing` |

### Telemetría (canaries)

| # | Métrica | Test |
|---|---|---|
| 40 | `stitching_ratio` = 1.0 con tracking limpio | `test_stitching_ratio_1_when_each_track_emits_once` |
| 41 | `stitching_ratio` > 1 detecta fragmentación | `test_stitching_ratio_detects_fragmentation` |
| 42 | `stitching_ratio` = 0 sin counts | `test_stitching_ratio_zero_when_no_counts` |
| 43 | `reset_daily` limpia stitching state | `test_reset_daily_clears_stitching_state` |
| 44 | `death_emit_count` incrementa solo en emit real | `test_death_emit_count_incremented_only_on_actual_emit` |
| 45 | `reset_daily` limpia death_emit_count | `test_reset_daily_clears_death_emit_count` |

### Gates anti-FP de emisión (ejes N × O × P × Q)

Los cuatro gates son **filtros de emisión ortogonales**: actúan sobre un track
que ya tiene cruce neto ≠ 0, decidiendo si el count se emite y/o si la
demografía se reporta. N/O/Q gatean el **conteo**; P gatea **solo la
demografía** (el count sale igual). Cada gate corre tanto en la rama de exit
observado como en el death-emit.

| # | Eje | Caso | Resultado | Test |
|---|---|---|---|---|
| 58 | N2 | altura mediana < `min_count_height_m` (exit) | no emit | `test_min_count_height_blocks_emit_for_short_track` |
| 59 | N3 | altura = None → gate no aplica | emit (recall preservado) | `test_min_count_height_passes_when_height_unknown` |
| 60 | N1 | altura ≥ umbral | emit | `test_min_count_height_passes_when_track_is_tall_enough` |
| 61 | N2 | altura < umbral en death-emit | no emit | `test_min_count_height_blocks_death_emit_for_short_track` |
| 62 | N off | default `0.0` desactiva el gate | emit | `test_min_count_height_default_is_off`, `test_build_counter_reads_min_count_height_m_from_config` |
| 63 | O2 | sin altura + conf < `min_count_confidence` (perro) | no emit | `test_min_count_confidence_blocks_dog_no_height` (parametrizado) |
| 64 | O1 | sin altura + conf alta (persona) | emit (recall preservado) | `test_min_count_confidence_passes_no_height_high_conf` |
| 65 | O3 | altura presente → gate de conf ignorado | emit | `test_min_count_confidence_ignored_when_height_present` |
| 66 | O2 | sin altura + conf baja en death-emit | no emit | `test_min_count_confidence_blocks_death_emit_dog` |
| 67 | O off | `0.0` desactiva; default es `0.60` | emit / config | `test_min_count_confidence_off_when_zero`, `test_min_count_confidence_default_is_060`, `test_build_counter_reads_min_count_confidence_from_config` |
| 68 | P2 | conf < `height_confidence_gate` → demografía unknown, **count sale** | emit sin demografía | `test_height_confidence_gate_below_threshold_marks_unknown` |
| 69 | P1 | conf ≥ umbral → reporta demografía | emit con demografía | `test_height_confidence_gate_above_threshold_reports_demographics` |
| 70 | P default/override | default = constante de clase; override por constructor | — | `test_height_confidence_gate_default_from_class_constant`, `test_height_confidence_gate_override_via_constructor`, `test_build_counter_reads_height_confidence_gate_from_config` |
| 71 | Q2 | `real_inside_frames` < `min_real_inside_frames` (single-frame flicker, exit) | no emit | `test_min_real_inside_frames_blocks_single_frame_entry` |
| 72 | Q1 | caminante con suficientes frames reales | emit | `test_min_real_inside_frames_passes_walker_with_enough_frames` |
| 73 | Q2 | evidencia fina en death-emit | no emit | `test_min_real_inside_frames_blocks_death_emit_thin_evidence` |
| 74 | Q off | default `0` desactiva el gate | emit / config | `test_min_real_inside_frames_default_is_off`, `test_build_counter_reads_min_real_inside_frames_from_config` |

**Cobertura completa** — los 4 gates tienen tests para: rama exit, rama
death-emit, default off/on, lectura desde config, y los casos de
recall-preservado (N3 altura None, O1 conf alta sin altura). No quedan gaps.

### Cross-cutting con tracker

| # | Eje | Caso | Test |
|---|---|---|---|
| 46 | K1 | same-id full visit | implícito en todos los `test_ingress_*` |
| 47 | K2 + M1 | adoption con outside_pos preservado | `test_ghost_adoption_preserves_close_outside_pos` |
| 48 | K2 + M2 | adoption con outside_pos invalidado (dist > 150) | `test_ghost_adoption_invalidates_far_outside_pos` |
| 49 | K3 | low IoU → no adoption | `test_ghost_pool_rejects_low_iou_match` |
| 50 | K3 | ventana expirada → no adoption | `test_ghost_pool_expires_after_window` |
| 51 | L1 | keep-alive mantiene PENDING inside | `test_keepalive_counting_zone_keeps_pending_alive_inside`, `*_recovers_to_confirmed_after_long_gap`, `*_extrapolates_does_not_freeze`, `*_capped_orphan_dies` |
| 52 | L2 | keep-alive no protege outside | `test_keepalive_counting_zone_does_not_protect_outside` |
| 53 | Lowe ratio | rechaza match ambiguo | `test_pass2_recovers_confirmed_with_bbox_jitter`, `test_pass2_respects_depth_gate`, `test_pass2_does_not_misroute_when_pass1_was_clean` |
| 54 | Knob ghost outside cap | configurable per-instancia | `test_ghost_outside_invalidate_px_is_configurable` |

### E2E (integración pipeline)

| # | Escenario | Test |
|---|---|---|
| 55 | Ingress E2E con MQTT mock | `test_e2e_ingress_event_published` |
| 56 | Indeciso E2E (entra y vuelve, no cuenta) | `test_e2e_indeciso_no_count` |
| 57 | Telemetría dispara en intervalo | `test_e2e_telemetry_fires_on_interval` |

---

## Celdas justificadas como structurally void

Combinaciones que el código **no puede alcanzar** o **no tiene sentido semántico**, documentadas para evitar tests defensivos sin valor:

- **A2 × cualquier B/C/D/E**: A2 (entry-Kalman) hace short-circuit con `return None` antes de tocar las otras dimensiones. Cubierto por #28-29.
- **J3 × cualquier B/C/D/E**: el counter rechaza CANDIDATE en la primera línea de `_process_track`. Cubierto por #32.
- **B4 × C1/C2/C3/C5 × D/E**: B4 implica `not any(n != 0)`, lo que significa `net=0` y por construcción `C4`. Las otras C son contradictorias.
- **B5 × D2**: si un track muere para ser ghost-adoptable, normalmente ya tuvo evidencia outside (lead-in). Un track born-inside que muere y es adoptado es teóricamente posible pero geometricamente raro y no agrega bug surface vs B3+D2 (que sí está cubierto).
- **I2 × ortogonalidad completa**: la orientación vertical es simetría espejo de la horizontal en `Line.side_of`, `within_segment` y `crossing_label`. Los tests `test_vertical_line_ingress/egress` validan la simetría — replicar todos los cuadrantes A×B×C×D×E para vertical sería ceremonia.
- **G2 × cada celda individual**: multi-line acumula nets independientes (validado en #8, gap-fill `test_multi_line_each_line_tracks_independent_net`). El comportamiento ortogonal de cada línea individual está cubierto en G1.
- **H1 ↔ H2**: el sticky de one-way es responsabilidad de `Line.crossing_label` (returna `None` para dirección sin label). Cubierto por #33 + invariante de `Line` testeado en `test_line_*`.

---

## Gap-fills agregados en este pase

Tres tests escritos para celdas significativas que el suite no cubría:

| Test | Celda | Por qué importa |
|---|---|---|
| `test_death_with_zigzag_net_zero_does_not_emit` | A1+B3+C3 | Valida el short-circuit `not any(n != 0)` en death-emit (equivalente al `test_indeciso_with_two_way_line_cancels` pero por muerte). Sin él, un zigzag-y-muere podría regresar y emitir doble. |
| `test_real_cycle_with_double_crossing_same_direction_counts_once` | A1+B1+C5 | Valida el invariante "verdict por signo del net, no magnitud". Sin él, una regresión que sumara el magnitude crearía counts duplicados. |
| `test_multi_line_each_line_tracks_independent_net` | G2 cross-line | Valida que el net de cada línea es independiente (no se contamina cross-line). Cubre el escenario de gates compuestos. |
| `test_ghost_adoption_preserves_counter_meta_so_resurrected_track_emits` | K2 end-to-end | Valida el **contrato counter↔tracker** que justifica la capa 1 del rescue cascade. El test anterior solo probaba que el meta arbitrario se preserva — este valida que el meta DEL COUNTER (crossing_net, inside, had_outside_pos) habilita que el track resucitado emita correctamente al salir. |

---

## Mantenimiento

Cuando se agregue una bifurcación nueva al counter o al tracker:

1. Agregar el eje (si es nuevo) o el valor (si es valor nuevo de un eje existente) a la tabla de **dimensiones discriminantes**.
2. Mappear el o los tests que cubren los nuevos casos en el cuadrante correspondiente.
3. Si la combinación introduce una celda nueva sin test → escribir el gap-fill antes de commitear el cambio del código.
4. Si la combinación nueva es structurally void → agregarla a la sección **void** con justificación.

El matrix es un doc vivo. Su valor es proporcional a qué tan honesto sea
el mapeo entre lo que el código discrimina y los tests que lo cubren.

Para la filosofía operacional + runbook de tuning (qué pasa cuando un site
diverge del baseline), ver [`docs/tracker_tuning.md`](tracker_tuning.md).
