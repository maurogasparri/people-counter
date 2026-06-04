-- Objetivos de negocio por métrica — fuente única de verdad para el semáforo del
-- ranking (②) y la comparativa "vs objetivo" (③). Editable con UPDATE sin redeploy
-- de dashboards; las queries leen estos valores en vivo. Mismo espíritu que
-- height_class()/rssi_class(): config en la DB, no hardcodeada.
--
-- direction: 'higher_better' (conversión, upt, ticket, ventas/m²) o 'lower_better'
-- (devoluciones — el objetivo es un MÁXIMO; cumplir = estar por debajo).
-- target: en la MISMA unidad cruda que la métrica (conversión/devolución como
-- fracción 0-1; ticket/ventas-m² en pesos; upt adimensional).

CREATE TABLE IF NOT EXISTS metric_targets (
    metric      TEXT PRIMARY KEY,
    target      NUMERIC NOT NULL,
    direction   TEXT NOT NULL DEFAULT 'higher_better'
                CHECK (direction IN ('higher_better', 'lower_better')),
    label       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO metric_targets (metric, target, direction, label) VALUES
    ('conversion',    0.05,   'higher_better', 'Tasa de conversión'),
    ('upt',           1.4,    'higher_better', 'Unidades por ticket'),
    ('ticket',        165000, 'higher_better', 'Ticket promedio'),
    ('sales_per_sqm', 53000,  'higher_better', 'Facturación por m²'),
    ('return_rate',   0.02,   'lower_better',  'Tasa de devolución')
ON CONFLICT (metric) DO NOTHING;

-- Lectura para Grafana (vistas que ya usan readonly_external la pueden necesitar):
GRANT SELECT ON metric_targets TO readonly_external;
