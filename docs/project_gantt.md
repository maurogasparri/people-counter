# Cronograma del proyecto — People Counter

Vista derivada del [plan de sprints](plan_sprints.md) (artefactos Scrum). El
cronograma sobre calendario —los doce sprints semanales ubicados en fechas, junto
con el bloque académico— se presenta como diagrama de Gantt en el TFG. Este
documento consolida la **secuencia**, el **esfuerzo por sprint** y las
**dependencias** que ese calendario materializa.

**Marco:** 12 sprints semanales secuenciales (S1 → S12), un único *Sprint Goal*
por sprint. **Esfuerzo total estimado: ~240 h.**

---

## Esfuerzo por sprint

| Sprint | Meta | Horas | Carga |
|:---:|---|---:|:---:|
| S1 | Análisis y diseño inicial | 16 | bajo |
| S2 | Captura estéreo y servicios | 20 | medio |
| S3 | Calibración estéreo | 24 | alto |
| S4 | Profundidad y región de interés | 16 | bajo |
| S5 | Detección neuronal de personas | 24 | alto |
| S6 | Seguimiento y conteo | 24 | alto |
| S7 | Captura WiFi y BLE | 20 | medio |
| S8 | Mensajería y telemetría | 20 | medio |
| S9 | Servicios cloud y APIs | 20 | medio |
| S10 | Visualización analítica | 16 | bajo |
| S11 | Validación y documentación | 24 | alto |
| S12 | Cierre del prototipo | 16 | bajo |
| | **Total** | **240** | |

---

## Secuencia y dependencias

El proyecto tiene dos cadenas de trabajo que avanzan en paralelo y convergen en la
nube:

- **Cadena de visión (camino crítico):**
  `S2 captura → S3 calibración → S4 profundidad → S5 detección → S6 seguimiento y conteo`.
  Cada eslabón depende del anterior: sin captura no hay calibración; sin
  calibración no hay profundidad; el conteo necesita el detector y la profundidad.
- **Cadena inalámbrica (paralela):** `S7 WiFi/BLE` es independiente de la cadena de
  visión y puede ejecutarse en paralelo; aporta su flujo de datos al transporte.
- **Convergencia en la nube:**
  `(S6 + S7) → S8 mensajería → S9 cloud → S10 visualización`.
  El transporte (S8) consume los eventos de ambas cadenas; la nube (S9) persiste y
  expone; la visualización (S10) cierra la mayoría de las historias de usuario al
  quedar disponible el dashboard.
- **Cierre:** `S11 validación` ejercita todo el sistema integrado y `S12` consolida
  el prototipo y los entregables.

S1 (análisis y diseño) precede a todo; el camino crítico que gobierna la duración
es la cadena de visión seguida de la convergencia cloud
(`S1 → S2 → S3 → S4 → S5 → S6 → S8 → S9 → S10 → S11 → S12`), con S7 absorbido en
paralelo.

---

## Hitos (puertas de avance)

Hitos planificados como criterio de cierre incremental, alineados al fin de cada
sprint (sin fecha — el calendario vive en el TFG):

| Hito | Criterio | Sprint |
|---|---|:---:|
| M1 | Hardware ensamblado y verificado; captura estéreo E2E | S2 |
| M2 | Calibración estéreo aceptable (reproyección < 1 px) | S3 |
| M3 | Profundidad validada contra ground-truth | S4 |
| M4 | Detector fine-tuneado corriendo en Hailo | S5 |
| M5 | Conteo por cruce de línea E2E sobre frames | S6 |
| M6 | Stitching WiFi/BLE con ratio mensurable | S7 |
| M7 | MQTT + buffer local con reentrega sin pérdida | S8 |
| M8 | Stack cloud desplegado y verificado E2E | S9 |
| M9 | Dashboards + alerting disponibles (cierre de US funcionales) | S10 |
| M10 | Casos de prueba (TC-01…TC-19) ejecutados | S11 |
| M11 | Prototipo cerrado + entregables del TFG | S12 |

---

## Extensión post-PoC (fuera de las 240 h)

Rollout de flota (OTA + alta disponibilidad cloud), planificado fuera de los 12
sprints. Detalle en [`docs/ota_design.md`](ota_design.md) y en el anexo de
[`plan_sprints.md`](plan_sprints.md).
