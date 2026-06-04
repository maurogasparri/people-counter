# Dimensionamiento de hardware — RAM del dispositivo edge

Documento de soporte para la elección **Raspberry Pi 5 2GB** sobre las
variantes de 4GB / 8GB. La decisión es empírica: se midió el consumo real
del pipeline en piloto, se le aplicó un stress test agresivo simulando el
ambiente de 2GB, y se proyectó el costo por flota.

## Tl;dr

| Dimensión | Resultado |
|---|---|
| Working set del pipeline | ~270 MB sostenido, peak 281 MB bajo stress |
| RAM total usada del sistema | ~800 MB de los 2 GB |
| Headroom en Pi 5 2GB | ~1.2 GB libres |
| Tocó swap durante stress | **0 KB** escritos (`vmstat si=so=0`) |
| OOM events durante stress | 0 |
| Ahorro vs Pi 5 4GB | ~USD 10 / device |
| Decisión | **Pi 5 2GB** |

## Medición en operación normal

Pi 5 piloto corriendo el pipeline completo (visión + WiFi/BLE + MQTT +
status LED + preview web), 7h de uptime sostenido a las 10:30 ART del
2026-05-25:

```
service people-counter:  memory.current=290M  memory.peak=302M
sistema total:           used=799M  buff/cache=596M  available=7.1G
swap usage:              0 bytes (vmstat si=so=0 durante 7h)
OOM events:              ninguno en 7 días
load average:            4.47 / 3.72 / 3.33 (CPU saturado, no RAM)
```

Top procesos por RSS:

| Proceso | RSS |
|---|---|
| `python3` (pipeline) | 408 MB |
| `dbus-daemon` (BLE scanning vía BlueZ) | 114 MB |
| `NetworkManager` | 21 MB |
| `systemd` + `journal` + `udev` + otros | ~50 MB |

## Stress test agresivo

Para validar 2GB sin esperar el piloto natural, se simuló el ambiente
Pi 5 2GB en el hardware de 8GB físico:

**Setup**:

1. **Balloon de 6 GB** con `stress-ng --vm 1 --vm-bytes 6G --vm-keep`
   → deja ~1.0 GB available = equivalente a Pi 5 2GB.
2. **16 streams MJPEG concurrentes** al preview (`/stream`).
3. **8 loopers paralelos** golpeando `/health` (320 req/s en total).
4. **1 stressor de CPU** (3 de 4 cores saturados).
5. **Duración**: 5 minutos sostenidos.

**Resultados**:

| Métrica | Baseline | Peor en stress | Δ |
|---|---:|---:|---:|
| `service.memory.current` | 269 MB | 273 MB | +4 MB |
| `service.memory.peak` | 275 MB | **281 MB** | +6 MB |
| `cgroup memory.events.high` (soft cap trigger) | 0 | **0** | — |
| `cgroup memory.events.max` (hard cap trigger) | 0 | **0** | — |
| `cgroup memory.events.oom_kill` | 0 | **0** | — |
| Sistema available | 7.2 GB | **~1.0 GB** (simulado Pi 2GB) | — |
| Swap escrito (`vmstat so`) | 0 KB/s | **0 KB/s** | — |
| Swap reservado (no escrito) | 0 MB | pico 11 MB → 3 MB final | — |
| Load (4 cores) | 3.55 | **12.16** | +8.6 |

**Conclusiones**:

1. **El peak del service fue 281 MB**. Solo +6 MB sobre el baseline pese
   a 5 min de stress agresivo en ambiente artificialmente apretado.
2. **Cero throttles del cgroup**: `MemoryHigh=1G` y `MemoryMax=1.5G` están
   sobredimensionados 3.5× sobre el peor caso. Funcionan como circuit
   breakers defensivos sin riesgo de falsos positivos.
3. **Cero paging real**. El kernel reservó 11 MB como swappable en algún
   momento pero **nunca tocó la microSD** (vmstat `si=so=0`). El sysctl
   `vm.swappiness=10` cumple su rol.
4. **El bottleneck bajo carga extrema es CPU, no RAM**. Load llegó a 12
   sobre 4 cores; comprar 4GB no resuelve eso. Solo bajar resolución,
   cap fps o sparse stereo lo resolverían.

## Trade-offs explícitos

| Argumento para 2GB | Argumento para 4GB+ |
|---|---|
| Working set real ≈ 14% de 2 GB → headroom holgado | Margen para features futuras (multi-cam, sparse stereo, ML on-device extra) |
| Cero swap activity bajo stress sintético + 7h piloto | Si en producción aparece un corner case raro de leak / spike, 4GB lo absorbe |
| ~USD 10 / device de ahorro | 2-3% del costo total del kit (~USD 350) — no mueve la aguja |
| Demuestra dimensionamiento riguroso para defender el TFG | Más simple operacionalmente (no requiere knobs ni cgroup caps) |

**Cuándo reconsiderar 4GB**: si en piloto sostenido (24-48h, horario operativo
con tráfico real de WiFi/BLE de un mall lleno) se observa cualquier de:

- `service.memory.peak > 1 GB` (cap `MemoryHigh`).
- Eventos de `memory.events.high > 0` sostenidos en `systemctl status`.
- Bytes escritos a swap > 0 en `vmstat` (`so` columna).
- Cualquier OOM event en `journalctl -u people-counter.service`.

## Guardrails operacionales aplicados

Los tres knobs siguientes están en el repo y se aplican vía `setup_device.sh`
en cada device nuevo. Aplican a cualquier modelo de RAM (no solo 2GB) como
hardening defensivo:

### 1. Circuit breakers de memoria del cgroup

`config/people-counter.service`:

```ini
MemoryHigh=1G       # soft cap — kernel reclama paginas del cgroup primero
MemoryMax=1500M     # hard cap — OOM kill + Restart=always
```

Sized 3.5× sobre el peor caso observado. Si la app empieza a comerse RAM
por un memory leak hipotético, el cap dispara antes de que arrastre al
kernel + `dbus` + `NetworkManager` del sistema.

### 2. sysctl tuning

`config/sysctl-people-counter.conf` → `/etc/sysctl.d/99-people-counter.conf`:

```
vm.swappiness = 10              # prefiere descartar cache antes de paginar
vm.dirty_background_bytes = 16777216    # 16 MB - flush async temprano
vm.dirty_bytes = 67108864               # 64 MB - cap del sync writeback
```

`swappiness=10` evita que el kernel pagine páginas anónimas a la microSD
(lenta, mata latency de inferencia). Caps fijos del dirty pagecache evitan
bursts de fsync que bloquean I/O.

### 3. Audio stack masked

`pipewire`, `wireplumber` y `pipewire-pulse` están masked en
`setup_device.sh`. El sistema no usa audio (RGB LED es GPIO directo); libera
~30 MB sostenidos.

## Cómo verificar en cualquier device

```bash
# Memoria actual del service (incluye peak desde el último restart)
systemctl status people-counter.service | grep Memory

# Eventos del cgroup (deberían ser todos 0)
sudo cat /sys/fs/cgroup/system.slice/people-counter.service/memory.events

# Swap activity (si=so deberían ser 0)
vmstat 1 5

# Working set total del sistema
free -h
```

En un device sano, post-1h de uptime con tráfico normal:
- `Memory: ~270M (high: 1G, max: 1.4G, peak: <300M)`
- `memory.events`: todos `0`
- `vmstat si=so`: ambos `0`
- `free -h`: `used <1G` en una Pi 5 2GB

## Nota — sizing del RDS cloud (no edge)

Este documento es sobre el device edge. Un apunte aparte para el otro lado del
pipeline: el **RDS Postgres del PoC es `db.t4g.micro` (1 GB RAM)**. Dimensiona
bien para el **ingest streaming** del piloto (eventos de conteo en tiempo real
+ resúmenes WiFi/BLE cada 15 min + telemetría cada 5 min — carga baja y
constante).

**Lección operacional**: `db.t4g.micro` **OOM-ea en bulk-loads** — un INSERT
masivo (~1.5M filas en una sola transacción, ej. re-seed de demo o backfill) lo
tumba por falta de RAM. Mitigaciones:

- **Escalar temporalmente** a `db.t4g.small` (2 GB) durante el bulk-load y
  volver a `micro` después, o
- **Batchear** el load en transacciones más chicas (ej. 50k filas por commit).

El runbook detallado de este escenario (síntomas, comandos de scale-up/down,
recovery) vive en `cloud_dr.md`. Acá solo el pointer: si un bulk-load tira el
RDS, no es el sizing del edge — es esto.

## Histórico de la decisión

- **2026-05-25 10:30 ART**: snapshot piloto, working set 290 MB sostenido,
  0 swap usage en 7 días.
- **2026-05-25 11:00 ART**: aplicados los 3 guardrails (memory caps + sysctl
  + mask audio). Verificado en piloto.
- **2026-05-25 11:30 ART**: stress test agresivo de 5 min con balloon 6GB
  + 16 streams MJPEG + 8 loopers /health + CPU stressor. Peak del service:
  281 MB. Cero throttles, cero swap escrito, cero OOM.
- **2026-05-25 11:45 ART**: decisión confirmada — **Pi 5 2GB**.
