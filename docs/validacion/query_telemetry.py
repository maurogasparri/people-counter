#!/usr/bin/env python3
"""Snapshot de telemetria del device real + agregados 48h + test idempotencia.

- Ultima fila de telemetry (todos los campos HW: cpu/hailo temp, fan, power, fps...)
- Agregados 48h: min/avg/max de temps, power, fps; max throttled; canaries.
- TC-12 idempotencia: re-inserta un count_event existente ON CONFLICT DO NOTHING
  y verifica rowcount=0 (la UNIQUE garantiza dedup, sin contaminar datos).
Reproducible: py docs/validacion/query_telemetry.py
"""
import json
import boto3
import psycopg

EP = "people-counter-dev.cabqikscg4gm.us-east-1.rds.amazonaws.com"
SECRET = "people-counter/dev/rds-master"
DEV = "store-pilot-01-cam-01"


def main():
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    s = json.loads(sm.get_secret_value(SecretId=SECRET)["SecretString"])
    conn = psycopg.connect(host=EP, port=5432, dbname="people_counter",
                           user=s["username"], password=s["password"],
                           sslmode="require", connect_timeout=15)
    cur = conn.cursor()

    print("=== ULTIMA TELEMETRIA (device real) ===")
    cur.execute("SELECT * FROM telemetry WHERE device_id=%s "
                "ORDER BY event_ts DESC LIMIT 1", (DEV,))
    cols = [d.name for d in cur.description]
    row = cur.fetchone()
    if row:
        for k, v in zip(cols, row):
            if v is not None and k not in ("store_id",):
                print(f"  {k:<30} {v}")

    print("\n=== AGREGADOS 48h (device real) ===")
    cur.execute("""
      SELECT count(*) n,
        min(cpu_temp_c), avg(cpu_temp_c), max(cpu_temp_c),
        min(hailo_temp_c), avg(hailo_temp_c), max(hailo_temp_c),
        min(fan_rpm), avg(fan_rpm), max(fan_rpm),
        min(power_w), avg(power_w), max(power_w),
        min(fps), avg(fps), max(fps),
        max(throttled_flags), max(buffer_backlog_messages),
        max(mqtt_disconnect_count), max(service_restarts)
      FROM telemetry WHERE device_id=%s AND event_ts>=now()-INTERVAL '48 h'
    """, (DEV,))
    r = cur.fetchone()
    labels = ["n", "cpu_min", "cpu_avg", "cpu_max", "hailo_min", "hailo_avg",
              "hailo_max", "fan_min", "fan_avg", "fan_max", "pw_min", "pw_avg",
              "pw_max", "fps_min", "fps_avg", "fps_max", "throttled_max",
              "backlog_max", "mqtt_disc_max", "restarts_max"]
    for lab, v in zip(labels, r):
        if isinstance(v, float):
            v = round(v, 2)
        print(f"  {lab:<16} {v}")

    print("\n=== canaries (ultima fila) ===")
    cur.execute("""SELECT wifi_ble_stitching_ratio, track_stitching_ratio,
                   death_emit_count, ghost_adoption_count, clock_synchronized,
                   cam_left_ok, cam_right_ok, wifi_probe_ok, ble_scanner_ok,
                   fs_readonly, mqtt_connected
                   FROM telemetry WHERE device_id=%s
                   ORDER BY event_ts DESC LIMIT 1""", (DEV,))
    cn = cur.fetchone()
    for lab, v in zip(["wifi_ble_stitching", "track_stitching", "death_emit",
                       "ghost_adoption", "clock_sync", "cam_left_ok",
                       "cam_right_ok", "wifi_probe_ok", "ble_scanner_ok",
                       "fs_readonly", "mqtt_connected"], cn):
        print(f"  {lab:<20} {v}")

    print("\n=== TC-12 IDEMPOTENCIA (count_events) ===")
    cur.execute("SELECT device_id,event_ts,track_id,direction FROM count_events "
                "WHERE device_id=%s ORDER BY event_ts DESC LIMIT 1", (DEV,))
    ex = cur.fetchone()
    if ex:
        cur.execute("SELECT count(*) FROM count_events WHERE device_id=%s", (DEV,))
        before = cur.fetchone()[0]
        cur.execute("""INSERT INTO count_events
            (device_id,store_id,event_ts,track_id,direction)
            SELECT device_id,store_id,event_ts,track_id,direction
            FROM count_events
            WHERE device_id=%s AND event_ts=%s AND track_id=%s AND direction=%s
            ON CONFLICT DO NOTHING""",
            (ex[0], ex[1], ex[2], ex[3]))
        inserted = cur.rowcount
        conn.commit()
        cur.execute("SELECT count(*) FROM count_events WHERE device_id=%s", (DEV,))
        after = cur.fetchone()[0]
        print(f"  fila de prueba: {ex}")
        print(f"  count antes={before} | filas insertadas (re-insert dup)="
              f"{inserted} | count despues={after}")
        print(f"  VEREDICTO: {'PASS (idempotente, 0 dup)' if inserted==0 and before==after else 'FAIL'}")
    else:
        print("  sin count_events del device real para probar")
    conn.close()


if __name__ == "__main__":
    main()
