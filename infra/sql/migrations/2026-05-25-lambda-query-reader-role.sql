-- Migración: rol ``lambda_query_reader`` para la Lambda query_aggregates
-- (T9.12, US-09, RF-13). Auth IAM (rds_iam), SELECT-only sobre tablas raw.
--
-- Patrón idempotente DO $$ + DROP IF EXISTS para re-runs. Para nuevos
-- deploys, el rol ya está en bootstrap.sql.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lambda_query_reader') THEN
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM lambda_query_reader;
        REVOKE ALL ON SCHEMA public FROM lambda_query_reader;
        DROP USER lambda_query_reader;
    END IF;
END $$;

CREATE USER lambda_query_reader;
GRANT rds_iam TO lambda_query_reader;
GRANT USAGE ON SCHEMA public TO lambda_query_reader;
GRANT SELECT ON
    count_events, wifi_ble_summary, pos_transactions,
    sites, devices
TO lambda_query_reader;

COMMENT ON ROLE lambda_query_reader IS
    'Rol IAM-auth SELECT-only para la Lambda query_aggregates. Tercer role '
    'separado del lambda_writer y lambda_pos_writer por least privilege — '
    'la Lambda de consulta no necesita INSERT/UPDATE. Acceso a tablas raw '
    '(no solo views) porque el query agregga inline por bucket variable.';
