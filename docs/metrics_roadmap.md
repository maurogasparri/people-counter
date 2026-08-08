# Métricas — alcance y roadmap

Estado de las métricas de retail frente a los datos que el sistema captura hoy
(visión cenital monocam en puerta + WiFi/BLE pasivo + POS del cliente +
telemetría de device). Criterio: **no agregar funcionalidad nueva** salvo que
sea muy barata (un dato estático de config).

Las métricas servibles se calculan sobre la capa de rollup
(en `infra/sql/bootstrap.sql`): vistas `*_by_bucket_*`
rápidas (rollup + live tail ≤5s), consumidas por Grafana y por el read-only
externo. Ninguna requiere tabla de rollup nueva.

## ✅ Calculables (10) — implementadas o triviales sobre la fundación

| Métrica | Fórmula | Vista / dato |
|---|---|---|
| Dispositivos totales vs únicos | avistajes vs `COUNT(DISTINCT visitor_hash)` | `wifi_ble_by_bucket_*` (`visitors`) |
| Horas pico / estacionalidad | ingresos por hora/día/semana | `counting_by_bucket_hour/day` |
| Tasa de captura | ingresos ÷ passersby | `turn_in_rate_by_bucket_*` |
| Tiempo de permanencia | Ley de Little (ocupación / arribos) | `visit_duration_by_bucket_*` |
| Tasa de conversión | ventas ÷ ingresos | `conversion_by_bucket_*` |
| Ticket promedio (ATV) | facturación ÷ ventas | `pos_by_bucket_*` |
| UPT | unidades ÷ ventas | `pos_by_bucket_*` |
| Tasa de devoluciones | **unidades devueltas ÷ unidades vendidas** (`items_return ÷ items_sale`) | `pos_by_bucket_*` |
| Ventas por visitante (RPV) | facturación ÷ ingresos (= conversión × ticket) | `revenue_per_visitor_by_bucket_*` |
| Ventas por m² | facturación neta ÷ superficie | `sales_per_sqm_by_bucket_*` (col `sites.sales_area_m2`) |

**Caveats**: "dispositivos únicos" es intra-día (el `visitor_hash` rota a diario por
privacidad); "tiempo de permanencia" es estimación a nivel local (no por persona/zona).

## ❌ Fuera de alcance (16) — límites con causa, no gaps a cerrar

### Descartadas en diseño
| Métrica | Por qué |
|---|---|
| Atracción de vidriera | el único proxy posible (shoppers÷passersby por RSSI) no es preciso — "estar cerca" ≠ "frenar a mirar"; lo literal necesita visión exterior |
| Distribución por entrada | el dato ya está (`count_events.device_id`); con una sola cámara da 100%, se desagrega con 2+ cámaras por local |
| Ratio clientes/vendedor (STAR) | no existe un "staff típico" (varía por temporada, horario, día); requeriría feed de dotación por turno |

### Comportamiento in-store → visión multi-zona / no-video
`Mapa de calor por zonas`, `Conversión de probadores`, `Recorrido del cliente`,
`Tiempo y abandono de cola`. Hoy: monocam cenital en la puerta y **regla dura de
no transmitir video**. Requieren sensado multi-zona, fuera del alcance del prototipo.

### Inventario → feed de stock/costo del cliente
`GMROI`, `Sell-through`, `Merma (shrinkage)`. El POS manda transacciones (ventas),
no inventario, costo ni margen. Requieren integración con el sistema de stock.

### Identidad persistente → choque con privacidad por diseño
`Tasa de recurrentes`, `Frecuencia de visita`, `CLV`. El `visitor_hash` se rota
cada día (salt rotado + reset de grupos) → **no hay identidad cross-día por
diseño**. Es un límite explícito del producto, no un gap. Solo medimos
repetición intra-día (engagement, que sí está).

### Datos externos
`CAC` (gasto de marketing + identidad de cliente nuevo), `NPS` (encuestas),
`Productividad del personal` (horas trabajadas, feed de RRHH/scheduling).

## Arquitectura de servido (por qué escala)
- **Rollups incrementales por watermark** (`received_at`): cada refresh recomputa
  solo los buckets que recibieron datos nuevos → O(reciente), no O(historia).
- **Live tail**: el bucket abierto (hoy) se calcula en vivo desde el raw →
  el dato se refleja a los ≤5s (honra el claim edge→cloud), la historia es
  instantánea desde el rollup.
- **Sin triggers** en el hot-path de ingesta; auto-reparable (cada refresh
  recomputa desde la verdad). Detalle en la migración y en `CLAUDE.md`.
- **Dos fuentes para poblar los rollups**: (a) desde el raw, vía
  `refresh_rollups()` (el camino del pipeline en vivo); o (b) directo desde
  histórico YA AGREGADO del sistema anterior, vía
  [`infra/sql/migrate_historical_rollups.example.sql`](../infra/sql/migrate_historical_rollups.example.sql)
  (+ el loader `scripts/migrate_historical.py`), que inserta en las tablas base
  `rollup_*` sin pasar por el crudo. `refresh_rollups()` nunca pisa esos buckets
  (solo toca los que tienen eventos crudos nuevos).
