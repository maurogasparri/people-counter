# Plan de sprints — People Counter (artefactos Scrum)

**Proyecto:** Sistema de conteo de personas para retail
**Stack:** Raspberry Pi 5 + Hailo-8L + IMX708 estéreo + WiFi/BLE + AWS
**Marco de gestión:** Scrum Solo — 12 sprints semanales, una única meta (*Sprint Goal*) por sprint
**Modelo de trabajo:** Claude Code escribe el código (incluyendo tests) bajo dirección humana; el humano dirige, decide arquitectura y valida con hardware
**Esfuerzo total estimado:** ~240 horas

> Las horas son una **estimación de planificación** por sprint (banda 16–24 h,
> tres niveles de carga), no un registro contable preciso. Los *Product Backlog
> Items* concretos de cada sprint viven en el sprint backlog (sección 2). El
> cronograma sobre calendario es una derivación de este plan.

**Extensión post-PoC** (fuera de este plan): rollout de flota (OTA + HA cloud) — ver el anexo al final.

---

## 1. Plan de sprints (Sprint Goals)

Cada sprint tiene una única meta. Las metas se clasifican en **técnicas**
(*foundational*, sin historia de usuario directa: S1–S4 y S12) y **funcionales**
(materializan una o más historias de usuario: S5–S11). La columna *US habilitadas*
lista las historias cuya funcionalidad o fuente de datos **propia** se construye en
ese sprint —en negrita, las que además cierran ahí de punta a punta—; la
infraestructura de datos compartida (la cadena de visión y conteo y el transporte e
ingesta a la nube) habilita a todas las historias de analítica y no se atribuye a
una en particular.

| Sprint | Meta de sprint (*Sprint Goal*) | Horas | Prioridad | US habilitadas |
|:---:|---|---:|:---:|---|
| S1 | Análisis y diseño inicial | 16 | Alta | — |
| S2 | Captura estéreo y servicios | 20 | Alta | US-10 |
| S3 | Calibración estéreo | 24 | Alta | — |
| S4 | Profundidad y región de interés | 16 | Alta | — |
| S5 | Detección neuronal de personas | 24 | Alta | — |
| S6 | Seguimiento y conteo | 24 | Alta | US-05 |
| S7 | Captura WiFi y BLE | 20 | Alta | US-04 |
| S8 | Mensajería y telemetría | 20 | Alta | US-10, **US-11, US-12** |
| S9 | Servicios cloud y APIs | 20 | Alta | **US-06, US-08, US-09** |
| S10 | Visualización analítica | 16 | Alta | **US-01, US-02, US-03, US-04, US-05, US-07, US-10** |
| S11 | Validación y documentación | 24 | Alta | — |
| S12 | Cierre del prototipo | 16 | Media | — |
| | **Total** | **240** | | |

Niveles de carga: **alto (24 h)** = S3, S5, S6, S11 (calibración lab-intensiva,
*pipeline* de ML, tracking de mayor complejidad algorítmica, validación integral);
**medio (20 h)** = S2, S7, S8, S9; **bajo (16 h)** = S1, S4, S10, S12. La
prioridad de S12 es Media por comprender cierre y entregables parcialmente
absorbibles desde otros sprints.

---

## 2. Sprint backlog (*Product Backlog Items* por sprint)

El sprint backlog detalla, para cada sprint, los *Product Backlog Items* (PBI)
comprometidos para alcanzar su meta; el conjunto de estos ítems a lo largo de los
doce sprints constituye el *Product Backlog* completo del proyecto. Las actividades
transversales —documentación técnica, *cleanup*, *hardening* y tests automáticos—
se ejecutan en paralelo a la actividad principal de cada sprint.

| Sprint — Meta | Ítems principales (PBI) | Cierra |
|---|---|:---:|
| **S1 — Análisis y diseño inicial** | Especificaciones funcionales (RF-01 a RF-13) y no funcionales (RNF-01 a RNF-12); diseño general de arquitectura HW + SW + cloud; *bill of materials* y adquisición de componentes; setup del repositorio con herramientas de calidad configuradas —pytest como *quality gate*, más análisis estático y formateo— que se ejecutan localmente; la automatización de esa compuerta en integración continua no se implementó. Estructura inicial del proyecto. | — |
| **S2 — Captura estéreo y servicios** | Ensamblaje físico (RPi5 + AI HAT+ + cámaras IMX708 + PoE HAT + gabinete); *bootstrap* del sistema operativo; captura estéreo dual en modo *raw*; sistema de configuración con *back-compat*; status LED + *health monitor*; servicios systemd, *logrotate* y purga automática de *frames*; script de preparación del dispositivo; validación end-to-end de la captura sobre hardware. | — |
| **S3 — Calibración estéreo** | *Pipeline* ChArUco con modelo Kannala-Brandt para óptica *fisheye*; caracterización del sensor; asistentes de enfoque óptico, calibración guiada y *preview* rectificada; *rescale* analítico de calibraciones a otras resoluciones del mismo FOV; iteración con tablero físico hasta converger con error de reproyección < 1 px. | — |
| **S4 — Profundidad y región de interés** | *Pipeline* SGBM con post-filtro WLS; asistente de diagnóstico de profundidad; monitor de salud de la calibración por error epipolar; validación con escena real de laboratorio. | — |
| **S5 — Detección neuronal de personas** | Diseño del *approach* (cabeza y hombros sobre rectificación *inline*); captura *multi-site* del *dataset* y muestreo estratificado; *labeling* (convención cabeza+hombros) y conversión a formato YOLO; minería por aprendizaje activo; entrenamiento iterativo de YOLOv8n; evaluación contra conjunto de validación; compilación HEF e integración *runtime*; *bench* del detector sobre hardware. | — |
| **S6 — Seguimiento y conteo** | Filtro de Kalman; *tracker* con asociación en dos etapas estilo ByteTrack con *ghost pool*; *line crossing counter* con zona de conteo; selector interactivo de zona y línea virtual; conversión a coordenadas mundo y clasificación adulto/niño; supresión de detecciones sobre *clutter* estático; *live preview* HTTP/MJPEG con *overlay* de *tracks*; pruebas integrales del *pipeline* visual. | — |
| **S7 — Captura WiFi y BLE** | Análisis de la aleatorización de direcciones MAC; captura WiFi en modo monitor (*nexmon*) y BLE pasiva; seudonimización con *hash* SHA-256 truncado y sal diaria; *stitching* con cuatro reglas complementarias; *publisher* de resúmenes cada 15 min; servicio systemd dedicado; exportador con ofuscación visual para auditoría; validación en hardware con tráfico real. | — |
| **S8 — Mensajería y telemetría** | Cliente MQTT con TLS mutuo, reconexión y *backoff* exponencial; *buffer* SQLite local con retención de 72 h; Device *Shadow* operativo (aplicación en caliente de tres parámetros de negocio, reconciliación *reported*/*desired* y rechazo de valores inválidos); telemetría cada 5 min con *canary* de verificación del *shadow*; tests E2E con simulación de desconexión. | **US-11, US-12** |
| **S9 — Servicios cloud y APIs** | Diseño del *stack cloud*; CloudFormation phaseado (VPC, *security groups*, NAT); RDS PostgreSQL con *Point-in-Time Recovery*; ECS Fargate tras ALB con HTTPS; IoT Core con *Topic Rules* y certificados X.509; Lambda de persistencia con IAM mínimo; esquema SQL con vistas materializadas; acceso programático *readonly* con *whitelist* de IPs; API REST de *ingest* POS; API REST de consulta de agregados (SigV4, paginación por cursor, *ETag*, contrato OpenAPI 3.1); política de respaldos y DR. | **US-06, US-08, US-09** |
| **S10 — Visualización analítica** | Grafana sobre ECS Fargate con HTTPS; cinco tableros agrupados por audiencia (analítica comercial y operación/flota): panorama de la cadena, comparativa y *ranking* de sucursales, detalle por sucursal, patrones de afluencia y salud de la flota; exportación nativa a CSV/Excel; *alerting* por umbrales; *monitoring* CloudWatch. | **US-01, US-02, US-03, US-04, US-05, US-07, US-10** |
| **S11 — Validación y documentación** | Banco de pruebas en laboratorio; ejecución de los casos de prueba (TC-01 a TC-19); integración E2E y *smoke tests* del *stack* completo; *preflight* y verificación de hardware; *replay mode* del *pipeline*; consolidación de las guías técnicas; procedimiento de migración de datos históricos; endurecimiento del conteo frente a falsos positivos no humanos (filtro por zona, salvaguardas de altura y permanencia mínimas, *rescue cascade* configurable por sucursal con su manual de ajuste). | — |
| **S12 — Cierre del prototipo** | *Hardening* final del repositorio (*ruff sweeps*, *type hints*); *coverage* mínimo de tests; script de aprovisionamiento de nuevas unidades; dimensionamiento del despliegue como insumo del análisis económico; capturas de pantalla y entregables finales del Trabajo Final de Grado. | — |

Cada ítem corresponde a módulos o componentes del repositorio, lo que habilita la
trazabilidad del trabajo desde el plan hasta el código.

---

## 3. Matriz de trazabilidad de historias de usuario

Cada historia de usuario contra los sprints que construyen su funcionalidad o fuente
de datos **propia** y aquel donde cierra su valor (el primer sprint donde se ejerce
de punta a punta, **en negrita**). La infraestructura de datos compartida (la cadena
de visión y conteo y el transporte e ingesta a la nube) habilita a todas las
historias de analítica y no se repite en cada fila. La distribución del cierre es:
dos en S8, tres en S9 y siete en S10 (al quedar disponible la capa de visualización).

| US | Necesidad principal | Sprints que la construyen | Sprint de cierre |
|---|---|---|:---:|
| US-01 | Dashboard de tráfico en tiempo real | **S10** | S10 |
| US-02 | Alertas por umbral horario | **S10** | S10 |
| US-03 | Comparativa entre sucursales e histórico | **S10** | S10 |
| US-04 | Tasa de captación | S7 (captura WiFi/BLE), **S10** | S10 |
| US-05 | Segmentación adulto/niño | S6 (clasificación por rango de estatura), **S10** | S10 |
| US-06 | Tasa de conversión integrada con POS | **S9** | S9 |
| US-07 | Exportar reportes en formato CSV/Excel | **S10** | S10 |
| US-08 | Acceso programático a datos del sistema | **S9** | S9 |
| US-09 | API REST de consulta para integración externa | **S9** | S9 |
| US-10 | Monitorear salud de la flota | S2 (monitor de salud), S8 (telemetría), **S10** | S10 |
| US-11 | Buffer local y retransmisión | **S8** | S8 |
| US-12 | Configuración remota de parámetros operativos | **S8** | S8 |

---

## 4. Asunciones del modelo de estimación

- **Claude Code escribe el código** (incluyendo tests unitarios) bajo dirección del humano.
- **El humano dirige, revisa, decide arquitectura y valida con hardware.**
- Los bucles que dependen de hardware/datos físicos NO se aceleran con asistencia de IA:
  - Calibración con tablero ChArUco físico.
  - Captura del dataset de imágenes para training.
  - Tiempos de entrenamiento (con supervisión activa).
  - Validación end-to-end del PoC.
- El **labeling de dataset** y la **escritura de documentación técnica** tampoco se aceleran.
- Las decisiones de arquitectura (Fargate vs ECS, RDS vs Aurora, las 4 reglas de stitching) requieren tiempo humano de análisis.
- Bajo este modelo, las metas con mayor componente de trabajo manual no acelerable conservan un esfuerzo alto (S3, S5, S6, S11); las de mayor componente *boilerplate* o integración sobre SDKs maduros, un esfuerzo más acotado (S1, S4, S10, S12).

---

## Anexo — Extensión post-PoC: Rollout de flota (OTA + HA cloud)

**Fuera del plan de 12 sprints (no incluido en las 240 h).** Iniciativa post-PoC:
lo que vuelve el sistema operable a escala sin visitas a sitio. Diseño completo de
OTA en [`docs/ota_design.md`](ota_design.md): AWS IoT Jobs + S3 (presigned URL) +
firma Ed25519 + swap atómico A/B con rollback automático + canario observado en el
tablero de flota + ventana de horario cerrado. Junto con la alta disponibilidad
cloud (RDS Multi-AZ) del *backlog*.
