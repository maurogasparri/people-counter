"""Genera y carga datos MOCK en la RDS para desarrollo de dashboards de Grafana.

Crea N sucursales sintéticas (prefijo ``demo-``) con tráfico realista a lo largo
de un período configurable, poblando las 4 tablas raw (``count_events``,
``wifi_ble_events``, ``pos_transactions``, ``telemetry``) + las dimensiones
(``sites``/``devices``). Los datos respetan los invariantes que esperan las
vistas y los paneles:

  - Embudo coherente: passersby > shoppers > ingresos > ventas.
  - Patrón diario (horario 10-22) + semanal (picos vie/sáb) + más demanda
    después de las 16 h en días de semana.
  - Demografía adult/child/unknown (altura cruzando el threshold 1.55 m).
  - Dwell por visita -> ocupación que sube durante el día y drena al cierre
    (Ley de Little -> panel de duración de visita).
  - Visitantes WiFi/BLE en 1..N ventanas de 15 min (panel de engagement).
  - rssi_max distribuido en shopper (>=-55) / passerby (>=-75) / weak.
  - Ventas POS con mix de prendas (remera/jean) y ~1.4 unidades por ticket.

NO toca el piloto real: opera SOLO sobre store_ids con el prefijo dado. Por
default purga la data demo previa antes de re-sembrar (idempotente). El piloto
(``store-pilot-01``) nunca se ve afectado.

Conexión: reusa ``scripts.provision._rds_connect`` (master user vía Secrets
Manager + boto3). Requiere creds AWS + ``pip install '.[provisioning]'``.

Uso típico:
    py -3 scripts/seed_mock_data.py                     # spec por default
    py -3 scripts/seed_mock_data.py --dry-run            # solo imprime conteos
    py -3 scripts/seed_mock_data.py --purge-only         # borra la data demo
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

# Permite ``py -3 scripts/seed_mock_data.py`` (sin -m): mete el repo root en el
# path para poder importar ``scripts.provision``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_mock_data")

# Los timestamps se generan como INSTANTES UTC REALES en hora Argentina (UTC-3),
# igual que publica el device real. El bucketing local lo hace la capa de rollup
# vía sites.timezone (AT TIME ZONE), así cada tienda reporta en SU hora local y
# el sistema soporta sitios de distintos países. (Antes se guardaba "local-as-UTC"
# como atajo; se revirtió al hacer el bucketing tz-aware per-site.)
EVENT_TZ = timezone(timedelta(hours=-3))

# Período por default: desde 1-mar-2026 hasta HOY (dinámico) → demo "vivo" sin
# días futuros. Sin futuro evita barras adelantadas en los tableros (y el techo
# del live-tail ya blinda la perf si igual aparecieran). Re-seedear extiende
# automáticamente hasta la fecha de corrida.
DEFAULT_START = date(2026, 3, 1)
DEFAULT_END = date.today()

# --- Catálogo de sucursales demo ------------------------------------------------
# base_ins:  ingresos/día de referencia (el promedio real tras modular por día de
#            semana, que está normalizado a media 1.0). 3 de tráfico alto (~230) +
#            5 de tráfico medio (~120), con jitter para que no sean clones.
# conv:      tasa de conversión (ventas / ingresos), 4.5%-8%.
# turn_in:   tasa de captación (ingresos / passersby), 15%-25%. Define cuántos
#            visitantes WiFi/BLE sintetizar: passersby = ingresos / turn_in.
# dwell_med: mediana de permanencia en minutos (mean ≈ med * 1.44 con sigma 0.85).
# area_m2: superficie de venta (grandes 180-240, medianas 140-180) → ventas/m².
SITES = [
    # store_id   nombre                         lat        lon      base  conv  turn  dwell_med  area_m2
    # grandes (área 195-235, tráfico alto)
    ("demo-01", "Alto Palermo",               -34.5882, -58.4108, 245, 0.052, 0.18, 6.5, 220),
    ("demo-02", "Unicenter",                  -34.5106, -58.5245, 230, 0.046, 0.16, 7.0, 235),
    ("demo-03", "Córdoba Shopping",           -31.3795, -64.2200, 218, 0.068, 0.21, 6.0, 195),
    # medianas (área 145-175, tráfico medio)
    ("demo-04", "Abasto",                     -34.6037, -58.4109, 132, 0.058, 0.20, 6.0, 175),
    ("demo-05", "Alto Avellaneda",            -34.6680, -58.3705, 124, 0.049, 0.17, 5.5, 160),
    ("demo-06", "Alto Comahue",               -38.9489, -68.0668, 118, 0.075, 0.23, 6.5, 150),
    ("demo-07", "Mendoza Plaza",              -32.8990, -68.8000, 112, 0.063, 0.22, 5.5, 145),
    ("demo-08", "Los Gallegos",               -38.0008, -57.5540, 106, 0.080, 0.25, 7.0, 165),
]

# Horario 10-22; peso relativo de tráfico por hora (forma del día).
HOUR_WEIGHTS = {
    10: 0.35, 11: 0.50, 12: 0.70, 13: 0.80, 14: 0.70, 15: 0.65,
    16: 0.80, 17: 0.95, 18: 1.00, 19: 1.00, 20: 0.90, 21: 0.70, 22: 0.45,
}
OPEN_HOURS = sorted(HOUR_WEIGHTS)
CLOSE_TIME = time(22, 30)  # los egresos se clampean al cierre

# Días de semana hay más demanda después de las 16 h: re-pondera (no infla) el
# perfil diario hacia la tarde/noche en lun-vie.
WEEKDAY_EVENING_BOOST = 1.30

# Multiplicador por día de semana (lunes=0 .. domingo=6), normalizado a media 1.0
# para que ``base_ins`` sea el promedio diario real. Pico vie/sáb, domingo flojo.
WEEKDAY_MULT = [0.821, 0.772, 0.821, 0.917, 1.207, 1.497, 0.966]

# Demografía de los ingresos: adulto / niño / (resto = unknown).
ADULT_FRAC = 0.90
CHILD_FRAC = 0.07

# Dwell lognormal: mean ≈ mediana * exp(sigma^2/2) = mediana * 1.44.
DWELL_SIGMA = 0.85

# Pricing de prendas (ARS). Mix 60% remera / 40% jean, ~1.4 unidades por ticket.
PRICE_REMERA = 82000
PRICE_JEAN = 182000
P_REMERA = 0.60
ITEMS_CHOICES = [1, 2, 3, 4]
ITEMS_WEIGHTS = [0.67, 0.27, 0.05, 0.01]  # media ≈ 1.40
RETURN_RATE = 0.02  # devoluciones como fracción de las ventas

# Distribución de rssi_max de los visitantes WiFi/BLE (embudo monótono).
SHOPPER_FRAC = 0.30   # rssi >= -55  (tráfico cercano)
PASSERBY_FRAC = 0.45  # -75..-56     (pasa cerca); resto 0.25 = weak (< -75)
BLE_FRAC = 0.60       # protocolo: 60% BLE / 40% WiFi


def _daily_ins(rng: random.Random, base: float, day: date) -> float:
    return base * WEEKDAY_MULT[day.weekday()] * rng.uniform(0.88, 1.12)


def _bucket_ins_for_day(rng: random.Random, base: float, day: date) -> dict[int, int]:
    """Reparte los ingresos del día por hora (entero por hora abierta), con el
    boost de tarde en días de semana redistribuyendo (no inflando) el total."""
    daily = _daily_ins(rng, base, day)
    is_weekday = day.weekday() < 5
    weights = {
        h: HOUR_WEIGHTS[h] * (WEEKDAY_EVENING_BOOST if (is_weekday and h >= 16) else 1.0)
        for h in OPEN_HOURS
    }
    wsum = sum(weights.values())
    out: dict[int, int] = {}
    for h in OPEN_HOURS:
        share = daily * (weights[h] / wsum)
        out[h] = int(share) + (1 if rng.random() < (share - int(share)) else 0)
    return out


def _sample_height(rng: random.Random):
    u = rng.random()
    if u < ADULT_FRAC:
        return round(rng.uniform(1.58, 1.90), 2)
    if u < ADULT_FRAC + CHILD_FRAC:
        return round(rng.uniform(0.95, 1.45), 2)
    return None  # unknown -> height_class() = 'unknown'


def _rand_dt(rng: random.Random, day: date, hour: int) -> datetime:
    return datetime.combine(
        day, time(hour, rng.randint(0, 59), rng.randint(0, 59)), tzinfo=EVENT_TZ
    )


def gen_count_events(rng: random.Random, days, sites):
    """Cada ingreso genera su egreso desfasado por el dwell -> ocupación realista,
    ins == outs por día."""
    track = 0
    for (sid, _n, _la, _lo, base, _cv, _ti, dwell_med, _area) in sites:
        dev = f"{sid}-cam-01"
        mu = math.log(dwell_med)
        for day in days:
            close_dt = datetime.combine(day, CLOSE_TIME, tzinfo=EVENT_TZ)
            for hour, n in _bucket_ins_for_day(rng, base, day).items():
                for _ in range(n):
                    track += 1
                    t_in = _rand_dt(rng, day, hour)
                    h = _sample_height(rng)
                    conf = round(rng.uniform(0.55, 0.95), 2)
                    yield (dev, sid, t_in, "in", track, conf, h)
                    dmin = max(1.0, rng.lognormvariate(mu, DWELL_SIGMA))
                    t_out = t_in + timedelta(minutes=dmin)
                    if t_out > close_dt:
                        t_out = close_dt + timedelta(seconds=rng.randint(0, 1800))
                    # Garantiza egreso DESPUÉS del ingreso (un ingreso tardío,
                    # pasado el cierre, no debe egresar antes que él → rompería
                    # el cumsum de ocupación con valores negativos).
                    if t_out <= t_in:
                        t_out = t_in + timedelta(seconds=rng.randint(30, 120))
                    yield (dev, sid, t_out, "out", track, conf, h)


def gen_wifi_ble_events(rng: random.Random, days, sites):
    """Un visitor = 16 bytes random, aparece en 1..N ventanas de 15 min con
    rssi_max según su tier. n_visitors = ingresos / turn_in / (shopper+passerby)."""
    cls_sum = SHOPPER_FRAC + PASSERBY_FRAC
    for (sid, _n, _la, _lo, base, _cv, turn_in, _dw, _area) in sites:
        dev = f"{sid}-cam-01"
        for day in days:
            n_visitors = int(_daily_ins(rng, base, day) / turn_in / cls_sum)
            for _ in range(n_visitors):
                vhash = rng.randbytes(16)
                proto = "ble" if rng.random() < BLE_FRAC else "wifi"
                u = rng.random()
                if u < SHOPPER_FRAC:
                    rssi = rng.randint(-55, -35)
                elif u < cls_sum:
                    rssi = rng.randint(-75, -56)
                else:
                    rssi = rng.randint(-95, -76)
                r = rng.random()
                n_win = (
                    1 if r < 0.62 else 2 if r < 0.85
                    else rng.randint(3, 5) if r < 0.97 else rng.randint(6, 10)
                )
                hour = rng.choices(OPEN_HOURS, weights=[HOUR_WEIGHTS[h] for h in OPEN_HOURS])[0]
                base_dt = datetime.combine(day, time(hour, rng.randint(0, 59)), tzinfo=EVENT_TZ)
                w0 = base_dt.replace(minute=(base_dt.minute // 15) * 15, second=0, microsecond=0)
                for k in range(n_win):
                    ps = w0 + timedelta(minutes=15 * k)
                    pe = ps + timedelta(minutes=15)
                    fs = ps + timedelta(seconds=rng.randint(0, 400))
                    ls = min(fs + timedelta(seconds=rng.randint(10, 500)), pe - timedelta(seconds=1))
                    yield (dev, sid, vhash, proto, rssi, fs, ls, ps, pe)


def _ticket_amount_minor(rng: random.Random) -> tuple[int, int]:
    """Sortea (items, amount_minor) de un ticket: mix remera/jean, var ±8%."""
    items = rng.choices(ITEMS_CHOICES, weights=ITEMS_WEIGHTS)[0]
    total = 0.0
    for _ in range(items):
        price = PRICE_REMERA if rng.random() < P_REMERA else PRICE_JEAN
        total += price * rng.uniform(0.92, 1.08)
    return items, int(total * 100)


def gen_pos_transactions(rng: random.Random, days, sites):
    """Ventas (+ ~2% devoluciones) durante el día, ponderadas por el perfil diario."""
    methods = ["cash", "card", "qr", "card", "card"]
    # Factor de conversión MACRO compartido por día (±22%): clima, quincena,
    # promo, etc. afectan a TODA la cadena junta → el agregado de la flota
    # oscila día a día en vez de promediarse a una recta. Determinístico por
    # fecha (random.Random(ordinal)) para que sea estable entre corridas.
    day_factor = {d: random.Random(d.toordinal()).uniform(0.78, 1.22) for d in days}
    for (sid, _n, _la, _lo, base, conv, _ti, _dw, _area) in sites:
        seq = 0
        for day in days:
            daily_ins = _daily_ins(rng, base, day)
            # Conversión del día = base × factor macro (compartido) × jitter
            # micro per-store (±8%). El macro mueve la cadena entera; el micro
            # diferencia sucursales. Sin el macro, el agregado quedaba plano.
            daily_conv = conv * day_factor[day] * rng.uniform(0.92, 1.08)
            n_sales = int(daily_ins * daily_conv)
            # Bernoulli por venta (no round del total): con n_sales diario chico
            # por local (~6-20), int(round(n_sales*0.02)) caía a 0 casi siempre y
            # las devoluciones desaparecían. Así la tasa ~2% emerge en agregado.
            n_returns = sum(1 for _ in range(n_sales) if rng.random() < RETURN_RATE)
            for kind, count in (("sale", n_sales), ("return", n_returns)):
                for _ in range(count):
                    seq += 1
                    hour = rng.choices(OPEN_HOURS, weights=[HOUR_WEIGHTS[h] for h in OPEN_HOURS])[0]
                    ts = _rand_dt(rng, day, hour)
                    items, amount_minor = _ticket_amount_minor(rng)
                    if kind == "return":
                        items = min(items, 2)
                    txid = f"{sid}-{day.isoformat()}-{seq}"
                    yield (txid, sid, ts, kind, items, amount_minor, "ARS", rng.choice(methods))


def gen_telemetry(rng: random.Random, tel_days, sites):
    """Salud de device cada 15 min, sobre los días dados (anclados al presente)."""
    for (sid, *_rest) in sites:
        dev = f"{sid}-cam-01"
        cum_in = cum_out = 0
        for day in tel_days:
            # El device reporta salud 24/7 aunque la tienda esté cerrada; los
            # contadores acumulados sólo avanzan en horario de apertura (sin tráfico
            # fuera de OPEN_HOURS, pero CPU/Hailo/MQTT/disco se siguen enviando).
            for hour in range(0, 24):
                for minute in (0, 15, 30, 45):
                    ts = datetime.combine(day, time(hour, minute), tzinfo=EVENT_TZ)
                    if hour in HOUR_WEIGHTS:
                        cum_in += rng.randint(0, 6)
                        cum_out += rng.randint(0, 6)
                    yield (
                        dev, sid, ts,
                        round(rng.uniform(48, 64), 1),    # cpu_temp_c
                        round(rng.uniform(44, 58), 1),    # hailo_temp_c
                        rng.randint(38000, 52000),        # disk_free_mb
                        rng.randint(900, 1400),           # mem_available_mb
                        round(rng.gauss(27, 3), 1),       # fps
                        cum_in, cum_out,                  # total_in, total_out
                        rng.random() > 0.01,              # mqtt_connected
                        rng.random() > 0.02,              # wifi_probe_ok
                        rng.random() > 0.02,              # ble_scanner_ok
                        round(rng.uniform(0.85, 1.0), 3), # wifi_ble_stitching_ratio
                        round(rng.uniform(1.0, 1.25), 3), # track_stitching_ratio
                        rng.randint(0, 4),                # death_emit_count
                        rng.randint(0, 8),                # ghost_adoption_count
                    )


def _copy(cur, table, cols, rows):
    n = 0
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        for row in rows:
            cp.write_row(row)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack-name", default=os.environ.get("PC_STACK", "people-counter-dev"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--start", default=DEFAULT_START.isoformat(), help="Fecha inicio YYYY-MM-DD.")
    ap.add_argument("--end", default=DEFAULT_END.isoformat(), help="Fecha fin YYYY-MM-DD (inclusive).")
    ap.add_argument("--sites", type=int, default=len(SITES), help="Cuántas sucursales demo (max %d)." % len(SITES))
    ap.add_argument("--prefix", default="demo-", help="Prefijo de store_id demo (para purga selectiva).")
    ap.add_argument("--telemetry-days", type=int, default=14, help="Días recientes de telemetry (0 = ninguno). Anclados a hoy.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-purge", action="store_true", help="No borrar la data demo previa antes de sembrar.")
    ap.add_argument("--purge-only", action="store_true", help="Solo borrar la data demo y salir.")
    ap.add_argument("--dry-run", action="store_true", help="Generar y contar filas SIN insertar.")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    sites = SITES[: args.sites]
    days = [start + timedelta(days=d) for d in range((end - start).days + 1)]
    logger.info("Sucursales=%d  período=%s..%s (%d días)  prefijo=%s",
                len(sites), days[0], days[-1], len(days), args.prefix)

    # Telemetry anclada al presente real (no al futuro): últimos N días hasta hoy.
    today = datetime.now(EVENT_TZ).date()
    tel_end = min(end, today)
    tel_days = [tel_end - timedelta(days=d) for d in range(args.telemetry_days - 1, -1, -1)
                if start <= tel_end - timedelta(days=d) <= end] if args.telemetry_days else []

    if args.dry_run:
        c = sum(1 for _ in gen_count_events(random.Random(args.seed), days, sites))
        w = sum(1 for _ in gen_wifi_ble_events(random.Random(args.seed), days, sites))
        p = sum(1 for _ in gen_pos_transactions(random.Random(args.seed), days, sites))
        t = sum(1 for _ in gen_telemetry(random.Random(args.seed), tel_days, sites))
        logger.info("[dry-run] count=%d  wifi_ble=%d  pos=%d  telemetry=%d (%d días)  TOTAL=%d",
                    c, w, p, t, len(tel_days), c + w + p + t)
        return

    from scripts.provision import _rds_connect  # noqa: E402
    conn = _rds_connect(args.stack_name, args.region)
    try:
        with conn.cursor() as cur:
            like = args.prefix + "%"
            if not args.no_purge or args.purge_only:
                logger.info("Purgando data demo previa (store_id LIKE %r) — el piloto NO se toca…", like)
                for tbl in ("count_events", "wifi_ble_events", "pos_transactions", "telemetry"):
                    cur.execute(f"DELETE FROM {tbl} WHERE store_id LIKE %s", (like,))
                cur.execute("DELETE FROM devices WHERE store_id LIKE %s", (like,))
                cur.execute("DELETE FROM sites   WHERE store_id LIKE %s", (like,))
                if args.purge_only:
                    conn.commit()
                    logger.info("Purga completa. Salgo (--purge-only).")
                    return

            for (sid, name, lat, lon, _base, _cv, _ti, _dw, area_m2) in sites:
                cur.execute(
                    """INSERT INTO sites (store_id, store_name, latitude, longitude, timezone, address, sales_area_m2)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (store_id) DO UPDATE SET store_name=EXCLUDED.store_name,
                         latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude,
                         sales_area_m2=EXCLUDED.sales_area_m2, updated_at=now()""",
                    (sid, name, lat, lon, "America/Argentina/Buenos_Aires",
                     f"{name} 1000, Argentina", area_m2),
                )
                cur.execute(
                    """INSERT INTO devices (device_id, store_id, cam_label, firmware_version, installed_at)
                       VALUES (%s, %s, %s, %s, now())
                       ON CONFLICT (device_id) DO UPDATE SET store_id=EXCLUDED.store_id, updated_at=now()""",
                    (f"{sid}-cam-01", sid, "puerta principal", "demo-1.0"),
                )

            logger.info("Insertando count_events…")
            nc = _copy(cur, "count_events",
                       ("device_id", "store_id", "event_ts", "direction", "track_id", "confidence", "height_m"),
                       gen_count_events(random.Random(args.seed), days, sites))
            logger.info("  count_events: %d filas", nc)

            logger.info("Insertando wifi_ble_events…")
            nw = _copy(cur, "wifi_ble_events",
                       ("device_id", "store_id", "visitor_hash", "protocol", "rssi_max",
                        "first_seen_ts", "last_seen_ts", "period_start", "period_end"),
                       gen_wifi_ble_events(random.Random(args.seed), days, sites))
            logger.info("  wifi_ble_events: %d filas", nw)

            logger.info("Insertando pos_transactions…")
            np_ = _copy(cur, "pos_transactions",
                        ("transaction_id", "store_id", "event_ts", "type", "items",
                         "amount_minor", "currency", "payment_method"),
                        gen_pos_transactions(random.Random(args.seed), days, sites))
            logger.info("  pos_transactions: %d filas", np_)

            nt = 0
            if tel_days:
                logger.info("Insertando telemetry (%d días, %s..%s)…", len(tel_days), tel_days[0], tel_days[-1])
                nt = _copy(cur, "telemetry",
                           ("device_id", "store_id", "event_ts", "cpu_temp_c", "hailo_temp_c",
                            "disk_free_mb", "mem_available_mb", "fps", "total_in", "total_out",
                            "mqtt_connected", "wifi_probe_ok", "ble_scanner_ok",
                            "wifi_ble_stitching_ratio", "track_stitching_ratio",
                            "death_emit_count", "ghost_adoption_count"),
                           gen_telemetry(random.Random(args.seed), tel_days, sites))
                logger.info("  telemetry: %d filas", nt)

        conn.commit()
        logger.info("OK — commit. TOTAL insertado: %d filas en %d sucursales demo.",
                    nc + nw + np_ + nt, len(sites))
    except Exception:
        conn.rollback()
        logger.exception("FALLÓ — rollback, no se insertó nada.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
