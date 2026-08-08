# Privacidad — sistema de conteo de personas

Este documento describe el tratamiento de datos personales del sistema y
la salvaguarda técnica/organizativa que aplica cuando se activa la
captura opcional de frames "best-frame" para data de active learning.

> **Estado por defecto: OFF.** En la configuración por defecto del
> sistema (`config/config.example.yaml: best_frame.enabled: false`) **no
> se almacena ni se transmite ninguna imagen ni video**. El sistema procesa
> el frame en RAM, extrae metadatos (conteo, dirección, altura
> aproximada, confianza del detector) y descarta la imagen. Activar la
> captura local de un frame representativo por evento requiere los pasos
> documentados más abajo.

## Datos procesados

### Permanentes (se transmiten al cloud)

- Cantidad de cruces de la línea virtual por dirección, agregada por
  evento. No incluye identificación individual.
- Profundidad estimada de la cabeza (m), altura aproximada (m) y
  rango de estatura. Los literales `adult`/`child`/`unknown` del esquema
  designan tramos de altura, no edades. Métrica derivada por evento de
  conteo, *no* identificadora.
- Identificador opaco de visitante (`visitor_hash`): un valor aleatorio de
  **16 bytes** (`BYTEA` en RDS), transmitido como los **32 caracteres hex de un
  `uuid.uuid4().hex`**, que el dispositivo asigna a cada grupo de identidad
  post-stitching. **No deriva de la MAC ni del hash de la MAC**, y se renueva cada día (`reset_daily()` rota la salt y resetea
  los grupos). La MAC se hashea aparte con SHA-256 + salt local truncado a 16
  bytes **solo en el estado local del dispositivo** (`wifi_ble_dedup.sqlite`,
  salt rotada a diario); ese hash **nunca se transmite**. Nunca se almacenan ni
  transmiten MACs crudas.
- RSSI máximo crudo (`rssi_max`) por visitor/ventana. La categorización
  shopper/passerby/weak se aplica **server-side** (función SQL `rssi_class`),
  no en el dispositivo.
- Telemetría del dispositivo (temperaturas, FPS, uptime, cola del
  buffer MQTT) — no incluye datos del titular.

### Locales temporales (con `best_frame.enabled: true`)

- Un frame JPG por cada evento de cruce de línea, rectificado y con la
  cámara izquierda como referencia. El JPG vive **únicamente en disco
  local del dispositivo**, en `/var/lib/people-counter/best_frames/`.
- Se purga automáticamente a los `retention_days` días (default 7)
  mediante un timer de systemd (`people-counter-purge-best-frames.timer`)
  que invoca `scripts/purge_best_frames.py` diariamente a las 03:30.
- **Nunca** se transmite por MQTT ni se sube a AWS — ni la imagen ni el
  path. El payload del evento MQTT solo lleva metadatos de conteo
  (`direction`, `track_id`, `event_time`, `height_m`, `confidence`). El
  path del JPG aparece únicamente en los logs locales del dispositivo
  (`best_frame_saved ... path=...`), para acceso autorizado vía SSH.

## DPIA mini (Evaluación de Impacto)

| Aspecto | Detalle |
|---------|---------|
| Naturaleza del tratamiento | Captura de un frame JPG por evento de cruce |
| Alcance | Solo durante horario operativo del local; un frame por persona contada |
| Contexto | Local comercial con cartelería visible al ingreso |
| Finalidad | Mejora del modelo de detección (active learning) durante piloto |
| Base jurídica considerada | Interés legítimo del responsable, evaluado contra los derechos del titular en el contexto comercial. La determinación de la base jurídica aplicable corresponde al responsable del tratamiento y excede el alcance de este prototipo |
| Categorías de datos | Imagen visual no biométrica (no se ejecuta reconocimiento facial) |
| Categorías de titulares | Personas físicas que circulan por el local |
| Plazo de conservación | 7 días, enforced por timer + script |
| Destinatarios | Personal técnico autorizado del responsable; nunca se transmite a terceros sin ofuscación visual previa (`scripts/export_anonymized.py`) |
| Transferencias internacionales | Ninguna del frame; los metadatos agregados a AWS (us-east-1) ya están cubiertos por la política general |

### Medidas técnicas de mitigación

1. **Default OFF en código.** El default canónico en
   `config/config.example.yaml` define `best_frame.enabled: false`.
   Activar requiere editar explícitamente el override per-device en
   `/etc/people-counter/config.yaml` y el validador en
   `src/config/loader.py` chequea el tipo bool. No es accionable por
   error humano remoto.
2. **Local-only.** El cliente MQTT nunca recibe ni los bytes del JPG ni
   el path. Auditoría: revisar `src/main.py` en el loop "Publicar eventos
   de conteo" — el payload no contiene `best_frame_path`; el path solo se
   loguea localmente (`best_frame_saved`). `best_frame_mgr.commit()`
   escribe el JPG a disco y devuelve el path, que nunca entra al payload.
3. **Retención corta.** El timer corre todos los días, y el script
   borra cualquier archivo con `mtime` mayor a `retention_days`.
   `Persistent=true` en el timer recupera ejecuciones perdidas si el
   dispositivo estuvo apagado.
4. **Hardening systemd.** El servicio de purga usa
   `ProtectSystem=strict` y `ReadWritePaths` explícito a la carpeta de
   JPGs. No puede tocar otros archivos.
5. **Ofuscación visual antes de exportar.** Si los frames se envían fuera
   del dispositivo (e.g. para etiquetado externo), `scripts/export_anonymized.py`
   aplica blur al área del bbox conocido (o blur uniforme + canny edges
   como fallback si no hay metadata) antes de cualquier salida. Es una
   **ofuscación**, no una anonimización: el frame difuminado conserva
   silueta, vestimenta, acompañantes y contexto espacio-temporal.
6. **Sin reconocimiento facial.** El detector (YOLOv8n fine-tuneado,
   `people-counter-detector`) produce bounding boxes de cabeza top-down,
   no encodings faciales. No hay base de datos biométrica.

### Medidas organizativas

- Política de privacidad publicada y accesible, indicando claramente la
  presencia del sistema y los derechos del titular.
- Cartelería visible en el ingreso del local. Plantilla mínima:
  > **Sistema de conteo automático en uso.** Este establecimiento
  > utiliza un sistema de conteo de personas con visión por
  > computadora. No se realiza reconocimiento facial. Para más
  > información sobre el tratamiento de datos personales y sus
  > derechos, consulte: [URL_POLITICA_PRIVACIDAD] / [contacto del
  > responsable].
- Acceso por SSH al directorio de JPGs restringido al equipo técnico,
  con registro de sesiones.
- Procedimiento documentado para responder a solicitudes de los
  titulares (acceso, supresión) sobre los frames que pudieran
  contenerlos.

## Derechos del titular

Los marcos de referencia consultados durante el diseño (LPDP en Argentina,
RGPD en la UE) contemplan los siguientes derechos del titular. Su
aplicabilidad concreta y la evaluación de cumplimiento corresponden al
responsable del tratamiento:

- **Acceso**: solicitar confirmación sobre si sus datos están siendo
  tratados y obtener copia.
- **Rectificación**: solicitar la corrección de datos inexactos.
- **Supresión**: solicitar el borrado de sus datos cuando ya no sean
  necesarios o se haya retirado el consentimiento.
- **Oposición**: oponerse al tratamiento basado en interés legítimo.
- **Portabilidad**: recibir sus datos en formato estructurado.

Para ejercer estos derechos contactar al responsable del tratamiento
(ver política de privacidad). En la práctica los frames `best_frame`
viven 7 días, por lo que la supresión efectiva se logra esperando el
plazo, sin perjuicio de la solicitud explícita.

## Procedimiento para activar `best_frame.enabled: true`

> **No flippear esta toggle a true sin completar TODOS los siguientes
> pasos.** El default está pensado para que el sistema opere sin almacenar
> imágenes; activar la captura cambia la naturaleza del tratamiento y exige
> una evaluación propia por parte del responsable.

Checklist obligatorio:

- [ ] DPIA específico del piloto firmado por el responsable.
- [ ] Política de privacidad actualizada y publicada con los detalles
      del tratamiento (`best_frame_path`, retención, finalidad).
- [ ] Cartelería instalada en cada local del piloto, en posición
      visible al ingreso, con QR a la política.
- [ ] Procedimiento de respuesta a solicitudes de titular documentado
      e identificada la persona responsable de atenderlas.
- [ ] Validación legal del contrato con el operador del local
      (responsabilidad compartida o tratamiento por encargo según el
      caso).
- [ ] Verificación de que el timer de purga
      (`people-counter-purge-best-frames.timer`) está activo:
      `systemctl is-active people-counter-purge-best-frames.timer`.

Una vez completado, editar `/etc/people-counter/config.yaml` en el
device, agregar la sección `best_frame` con `enabled: true`, y reiniciar
el servicio (`systemctl restart people-counter`). La primera ejecución
del pipeline emitirá un `WARNING` en los logs confirmando la activación.

## Responsable del tratamiento

Completar antes del despliegue del piloto:

- **Nombre/Razón social**: _[a completar]_
- **Domicilio**: _[a completar]_
- **Contacto del DPO o responsable**: _[a completar]_
- **Política de privacidad**: _[URL a completar]_
- **Registro AAIP / autoridad de control**: _[a completar]_
