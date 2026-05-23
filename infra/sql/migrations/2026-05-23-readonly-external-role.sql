-- Migración: agregar el rol ``readonly_external`` (US-08, S9 T9.8).
-- Idempotente — usa el patrón DO $$ + DROP IF EXISTS para no fallar en
-- re-runs. Para nuevos deploys, el rol ya está en bootstrap.sql.
--
-- ⚠ El password placeholder lo cambia el operador después con:
--     ALTER USER readonly_external WITH PASSWORD '<password-fuerte>';
-- (ver docs/api_access.md). En producción conviene gestionarlo desde
-- Secrets Manager.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'readonly_external') THEN
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM readonly_external;
        REVOKE ALL ON SCHEMA public FROM readonly_external;
        DROP USER readonly_external;
    END IF;
END $$;

CREATE USER readonly_external WITH PASSWORD 'CHANGE_ME_FROM_SECRETS_MANAGER';
GRANT USAGE ON SCHEMA public TO readonly_external;
GRANT SELECT ON
    counting_hourly, counting_daily, counting_by_bucket,
    turn_in_rate_by_bucket,
    conversion_rate_hourly, conversion_rate_daily, conversion_rate_by_store,
    wifi_ble_store_traffic,
    sites, devices
TO readonly_external;

COMMENT ON ROLE readonly_external IS
    'Rol read-only para acceso programatico externo (partners, analistas). '
    'SELECT solo sobre views de agregados — no sobre tablas crudas. '
    'Auth via password (no IAM) — sincronizar el password desde Secrets '
    'Manager o ALTER USER manualmente. Conectividad requiere SG whitelist '
    '+ PubliclyAccessible=true en RDS (operator-managed).';
