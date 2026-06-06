# Diagramas del TFG — fuentes Mermaid

Borradores versionados de las figuras del documento, derivados de la fuente de
verdad del repo (código + `infra/`). Sirven como **guía exacta de actores,
mensajes, clases y relaciones** para redibujarlas en Visio (o renderizarlas
directo con Mermaid).

Cada figura lista su **fuente en el repo** para trazabilidad. Si el código
cambia, actualizar acá primero y re-exportar.

> **Render**: pegar el bloque en [mermaid.live](https://mermaid.live) o usar la
> extensión Mermaid de VS Code → exportar SVG/PNG. La Figura 9 (DER) ya está en
> [`database_schema.md`](database_schema.md) y no se duplica acá.

| Fig | Título | Tipo | Fuente en el repo |
|---|---|---|---|
| 4 | Evento de conteo desde captura estéreo | Secuencia | `src/main.py`, `src/vision/*`, `src/tracking/*`, `src/mqtt/*` |
| 5 | Resumen de tráfico desde captura WiFi/BLE | Secuencia | `src/wifi_ble/{wifi_probe,ble_scan,dedup,publisher}.py` |
| 6 | Integración y entrega de flujos a la nube | Secuencia | `src/mqtt/{client,buffer}.py`, `src/cloud/persist_event.py` |
| 7 | Arquitectura general (4 capas IoT) | Flujo / bloques | `README.md §Arquitectura`, `CLAUDE.md` |
| 8 | Despliegue del sistema | Despliegue | `infra/cloudformation/people-counter.yaml`, `CLAUDE.md §Hardware` |
| 9 | DER del esquema PostgreSQL | Entidad-relación | **[`database_schema.md`](database_schema.md)** (ya existe) |
| 10 | Modelo de clases de mensajería MQTT | Clases | `src/mqtt/{client,buffer}.py`, `src/wifi_ble/publisher.py` |
| 13 | Esquema de stitching de identidad inalámbrica | Conceptual / bloques | `CLAUDE.md §Captura WiFi/BLE`, `src/wifi_ble/{dedup,fingerprint}.py` |

---

## §3.4 — Diagramas de secuencia

### Figura 4 — Generación de un evento de conteo a partir de la captura estéreo

> **Fuente**: `src/main.py` (loop principal) → `src/vision/calibration.rectify_pair`
> → `src/vision/depth` (SGBM+WLS) → `src/vision/detect.detect_persons` (Hailo) →
> `src/tracking/tracker.EuclideanTracker.update` → `src/tracking/counter.Counter`
> → `src/mqtt/client.MQTTClient.publish_event` → `src/mqtt/buffer.MessageBuffer`.

```mermaid
sequenceDiagram
    autonumber
    participant CAM as Cámaras IMX708<br/>(StereoCapture)
    participant LOOP as Pipeline<br/>(main loop)
    participant CAL as Calibración<br/>(rectify_pair)
    participant DEP as Profundidad<br/>(SGBM + WLS)
    participant DET as Detector<br/>(HailoBackend)
    participant TRK as EuclideanTracker
    participant CNT as Counter
    participant MQ as MQTTClient
    participant BUF as MessageBuffer

    loop cada frame
        LOOP->>CAM: capture()
        CAM-->>LOOP: frame_l, frame_r
        LOOP->>CAL: rectify_pair(frame_l, frame_r)
        CAL-->>LOOP: rect_l, rect_r
        LOOP->>DEP: compute_depth(rect_l, rect_r)
        DEP-->>LOOP: depth_map (fresh o cache)
        LOOP->>DET: detect_persons(rect_l)
        DET-->>LOOP: detecciones[]
        LOOP->>TRK: update(posiciones, metas)
        TRK-->>LOOP: tracks (Kalman + state machine)
        LOOP->>CNT: update(tracks)
        alt cruce de línea (counting zone)
            CNT-->>LOOP: CountEvent(direction = in/out)
            LOOP->>MQ: publish_event("counting", payload)
            MQ->>BUF: enqueue(topic, payload)
            BUF-->>MQ: msg_id
            alt conectado
                MQ->>MQ: PUBLISH QoS1 a IoT Core
                Note over MQ,BUF: PUBACK → mark_sent(msg_id)
            else offline
                Note over MQ,BUF: queda en outbox para replay
            end
        else sin cruce
            CNT-->>LOOP: (sin evento)
        end
    end
```

### Figura 5 — Generación del resumen de tráfico a partir de la captura WiFi y BLE

> **Fuente**: `src/wifi_ble/wifi_probe.py` + `ble_scan.py` (productores en threads
> separados) → `src/wifi_ble/dedup.DedupEngine.process_detection` (hash+salt, 4
> reglas de stitching, bajo lock) → `src/wifi_ble/publisher.WifiBlePublisher.maybe_publish`
> (ventanas de 15 min) → `MQTTClient`.

```mermaid
sequenceDiagram
    autonumber
    participant WIFI as WiFiProbe<br/>(nexmon/scapy · thread)
    participant BLE as BLEScanner<br/>(bleak · thread)
    participant DED as DedupEngine<br/>(SQLite hash_groups)
    participant LOOP as Pipeline<br/>(main loop)
    participant PUB as WifiBlePublisher
    participant MQ as MQTTClient
    participant BUF as MessageBuffer

    par Captura WiFi
        WIFI->>WIFI: probe 802.11 capturado
        WIFI->>DED: process_detection(mac, rssi, seqnum, fingerprint)
        Note over DED: bajo lock · hash = SHA-256(mac + salt local)<br/>4 reglas de stitching → group_id · upsert hash_groups
    and Captura BLE
        BLE->>BLE: advertisement (RPA) capturado
        BLE->>DED: process_detection(addr, rssi, fingerprint)
        Note over DED: mismo lock/SQLite · stitching cross-protocol
    end

    loop cada tick del main loop
        LOOP->>PUB: maybe_publish()
        alt cruzó el boundary de 15 min
            PUB->>DED: get_window_records(period_start, period_end)
            DED-->>PUB: devices[] (1 por group_id · rssi_max crudo)
            alt ventana no vacía
                PUB->>MQ: publish_event("wifi_ble", {period_start, period_end, devices})
                MQ->>BUF: enqueue → IoT Core (QoS1) → PUBACK → mark_sent
            else ventana vacía
                PUB-->>LOOP: 0 (no publica)
            end
        else aún no toca
            PUB-->>LOOP: 0
        end
    end
```

### Figura 6 — Integración y entrega de los flujos hacia la nube

> **Fuente**: `src/mqtt/client.MQTTClient` (publish + replay + PUBACK) +
> `src/mqtt/buffer.MessageBuffer` (outbox) → AWS IoT Core (3 Topic Rules) →
> `src/cloud/persist_event.handler` (IAM token, dispatch por tipo, INSERT
> idempotente) → RDS Postgres.

```mermaid
sequenceDiagram
    autonumber
    participant MQ as MQTTClient<br/>(device)
    participant BUF as MessageBuffer<br/>(outbox SQLite)
    participant IOT as AWS IoT Core<br/>(broker + Topic Rules)
    participant LAM as Lambda persist_event
    participant RDS as RDS Postgres 16

    Note over MQ,BUF: todo publish escribe el outbox ANTES de enviar
    MQ->>BUF: enqueue(topic, payload) → msg_id

    alt conectado
        MQ->>IOT: PUBLISH topic (QoS1 · TLS X.509)
        IOT->>IOT: Topic Rule SQL matchea por topic
        IOT->>LAM: invoke(envelope {device_id, timestamp, type, data})
        LAM->>LAM: _get_connection() · token IAM<br/>(rds.generate_db_auth_token)
        LAM->>RDS: INSERT ... ON CONFLICT (idempotente)
        alt inserción OK
            RDS-->>LAM: ok
            LAM-->>IOT: 200
            IOT-->>MQ: PUBACK
            MQ->>BUF: mark_sent(msg_id)
        else error transitorio (conn / token expirado)
            LAM-->>IOT: raise → IoT reintenta la regla
        else error de validación (JSON / constraint)
            LAM-->>IOT: 200 (descarta · sin reintento)
        end
    else offline
        Note over MQ,BUF: solo enqueue — el pipeline sigue corriendo
    end

    Note over MQ,IOT: al reconectar → _on_connect() dispara replay_buffer()
    MQ->>BUF: get_pending(limit=200)
    BUF-->>MQ: mensajes sent=0 (saltea los in-flight)
    MQ->>IOT: re-PUBLISH → PUBACK → mark_sent
```

---

## §3.5 — Arquitectura

### Figura 7 — Arquitectura general del sistema propuesto (las 4 capas IoT)

> **Fuente**: `README.md §Arquitectura` + `§Procesos en el edge`; diagrama de
> bloques de `CLAUDE.md`. Las 4 capas siguen el modelo de referencia IoT
> (percepción → red → procesamiento → aplicación), mapeado a los componentes
> reales del sistema.

```mermaid
flowchart TB
    subgraph L1["① Capa de DETECCIÓN (percepción / edge)"]
        direction LR
        CAM["2× Arducam IMX708<br/>par estéreo"]
        RADIO["WiFi/BLE monitor<br/>(probing pasivo)"]
        RPI["Raspberry Pi 5<br/>pipeline + dedup + outbox"]
        HAILO["Hailo-8L · YOLOv8n<br/>detección · tracking · conteo"]
        CAM --> RPI
        RADIO --> RPI
        RPI --> HAILO
    end

    subgraph L2["② Capa de INTERCAMBIO (red / transporte)"]
        MQTT["MQTT 3.1.1 · TLS + X.509 · QoS 1"]
        IOT["AWS IoT Core<br/>broker + 3 Topic Rules"]
        MQTT --> IOT
    end

    subgraph L3["③ Capa de PROCESAMIENTO (middleware / datos)"]
        LAM["Lambdas<br/>persist_event · ingest_pos · query_aggregates"]
        DB[("RDS Postgres 16<br/>raw + rollups + vistas")]
        LAM --> DB
    end

    subgraph L4["④ Capa de SERVICIOS (aplicación)"]
        GRAF["Grafana 13<br/>tableros: analítica + flota"]
        API["REST API<br/>socios / BI"]
    end

    POS["POS externo del cliente"]

    RPI -->|"solo metadatos<br/>(nunca video)"| MQTT
    IOT --> LAM
    POS -->|"HTTPS / API Gateway"| LAM
    DB --> GRAF
    DB --> API
```

### Figura 8 — Diagrama de despliegue del sistema

> **Fuente**: `infra/cloudformation/people-counter.yaml` (topología AWS completa)
> + `CLAUDE.md §Hardware` (nodo edge). Notación UML Deployment: **nodos** (cajas
> 3D en Visio), **artefactos** desplegados dentro, **conexiones** etiquetadas con
> protocolo.

```mermaid
flowchart TB
    OP["Operador / Analista<br/>(navegador)"]
    POS["POS externo<br/>del cliente"]

    subgraph DEV["«device» Raspberry Pi 5 — RPi OS Trixie"]
        direction TB
        A1["«artifact» pipeline src/ (main.py)"]
        A2["«artifact» modelo HEF (Hailo-8L)"]
        A3["«artifact» certs X.509"]
        A4["«artifact» outbox + dedup (SQLite)"]
        HW["HW: Hailo-8L · 2× IMX708 · PoE HAT · LED RGB"]
    end

    subgraph AWS["«executionEnvironment» AWS Cloud (us-east-1)"]
        direction TB
        IOT["«node» IoT Core<br/>broker + 3 Topic Rules"]
        subgraph LAMS["«node» Lambda (fuera de VPC · IAM auth)"]
            L1["persist_event"]
            L2["ingest_pos_transaction"]
            L3["query_aggregates"]
        end
        RDS[("«node» RDS Postgres 16<br/>db.t4g.micro · force_ssl")]
        subgraph ECS["«node» ECS Fargate + ALB + ACM"]
            GRAF["«artifact» Grafana 13<br/>(imagen desde ECR)"]
        end
    end

    DEV -->|"MQTT 8883<br/>TLS X.509 · QoS1"| IOT
    IOT --> L1
    L1 -->|"IAM token"| RDS
    POS -->|"HTTPS<br/>(API Gateway)"| L2
    L2 --> RDS
    L3 --> RDS
    OP -->|"HTTPS 443<br/>(dominio + ACM)"| GRAF
    GRAF -->|"datasource SQL"| RDS
```

---

## §3.6 — Estructura de datos

### Figura 9 — DER del esquema en PostgreSQL

Ver **[`database_schema.md`](database_schema.md)** — bloque `erDiagram` con las 4
tablas de hechos (`count_events`, `wifi_ble_events`, `telemetry`,
`pos_transactions`), las 3 dimensiones (`sites`, `devices`, `holidays`) y la única
FK real (`devices → sites`). Es la fuente canónica del DER; no se duplica acá.

### Figura 10 — Diagrama de clases del modelo de mensajería MQTT

> **Fuente**: `src/mqtt/client.MQTTClient`, `src/mqtt/buffer.MessageBuffer`,
> `src/wifi_ble/publisher.WifiBlePublisher` + sus Protocols. Acotado a la capa de
> mensajería (no todo el sistema). El envelope estándar (`docs/api_contracts.md`)
> va como tipo de dato.

```mermaid
classDiagram
    class MQTTClient {
        +str device_id
        +str endpoint
        +int port
        +dict topics
        +bool connected
        +int disconnect_count
        -MessageBuffer buffer
        -dict _pending_acks
        +connect(startup_jitter_seconds)
        +publish(topic, payload, qos) int
        +publish_event(event_type, data, qos) int
        +replay_buffer() int
        +subscribe_shadow_delta(thing, cb)
        +publish_shadow_reported(thing, state)
        -_on_connect()
        -_on_publish() : PUBACK → mark_sent
        -_on_message() : shadow delta
    }

    class MessageBuffer {
        +str db_path
        +int max_age_hours
        +int max_backlog
        +enqueue(topic, payload) int
        +mark_sent(message_id) bool
        +get_pending(limit) list
        +purge_old() int
        +enforce_backlog_limit() int
        +count_unsent() int
    }

    class WifiBlePublisher {
        -_MQTTPublisher _mqtt
        -_DedupRecords _dedup
        -float _period
        +maybe_publish() int
        +last_period_end() float
    }

    class _MQTTPublisher {
        <<interface>>
        +publish_event(event_type, data, qos) int
    }

    class _DedupRecords {
        <<interface>>
        +get_window_records(since_ts, until_ts) list
    }

    class _NullMQTTClient {
        +publish_event(...) None
    }

    class Envelope {
        <<dict estándar>>
        +str device_id
        +float timestamp
        +str type
        +dict data
    }

    MQTTClient *-- MessageBuffer : compone (outbox)
    MQTTClient ..|> _MQTTPublisher : satisface
    _NullMQTTClient ..|> _MQTTPublisher : satisface
    WifiBlePublisher ..> _MQTTPublisher : usa
    WifiBlePublisher ..> _DedupRecords : usa
    MQTTClient ..> Envelope : publish_event() arma
```

---

## §4.4 — Captura WiFi y BLE

### Figura 13 — Esquema de stitching de identidad inalámbrica

> **Fuente**: `CLAUDE.md §Captura WiFi/BLE` (las 4 reglas) +
> `src/wifi_ble/dedup.py` + `fingerprint.py`. Es un **esquema conceptual** (no UML
> estándar): muestra cómo identidades inalámbricas rotativas (MAC randomizada /
> RPA BLE) se correlacionan en un `group_id` estable antes de contarse.

```mermaid
flowchart TB
    subgraph IN["Identidades observadas (rotativas)"]
        W1["WiFi MAC #1<br/>seqnum, RSSI, fingerprint"]
        W2["WiFi MAC #2<br/>(rotó a los ~2 min)"]
        B1["BLE RPA<br/>AddressType=random"]
    end

    FILT{"¿randomizada / random?<br/>(filtro de humanos)"}
    W1 --> FILT
    W2 --> FILT
    B1 --> FILT
    FILT -->|"global / public<br/>(infra fija)"| DROP["descartado<br/>(no se hashea)"]

    FILT -->|"sí"| HASH["hash = SHA-256(addr + salt local)<br/>truncado a 16 bytes"]

    subgraph RULES["4 reglas de stitching (joina al grupo más reciente que matchee)"]
        direction TB
        R1["① Seqnum continuity (WiFi)<br/>Δseqnum ≤ 100 · ΔRSSI ≤ 5 · Δt ≤ 30s"]
        R2["② Cross-protocol L2 (≤ 2s)<br/>WiFi MAC ↔ BLE addr · ΔRSSI ≤ 5"]
        R3["③ BLE anchoring (~15 min)<br/>nuevas MAC al grupo del BLE vivo"]
        R4["④ Fingerprint continuity<br/>IEs/HT/VHT (WiFi) · Continuity (BLE)"]
    end
    HASH --> RULES

    RULES --> GID["group_id estable<br/>(UUID random · inlinkeable)"]
    GID --> STORE[("hash_groups (SQLite local)<br/>rotado diario + salt rotado")]

    GID --> COUNT["conteo: DISTINCT group_id<br/>passersby / shoppers (rssi_class server-side)"]
    COUNT --> PUB["MQTT: solo group_id opaco + rssi_max crudo<br/>(nunca MAC, seqnum ni hash pre-stitch)"]
```
