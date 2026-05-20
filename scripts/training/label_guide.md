# Guía de labeling — detector cenital de personas

Convención canónica para etiquetar el dataset del detector. Mantener esto
consistente entre sesiones es lo que evita el drift (descubrimos tarde que el
v1 etiquetaba silueta completa en vez de cabeza+hombros). Leer antes de cada
sesión de labeling.

## Qué se etiqueta

- **Una sola clase: `person`.** No hay otras clases. (El bbox es de
  cabeza+hombros, pero la clase se llama `person` porque contamos personas —
  NO la llames "head".)
- **Forma: bounding box (rectángulo) de CABEZA + HOMBROS.** Desde vista
  cenital una persona se ve como una "T": la cabeza (mancha redonda) + la
  línea de los hombros. El bbox envuelve eso.
  - **NO** la silueta completa (no incluir torso bajo, brazos extendidos,
    piernas, ni la sombra en el piso).
  - **NO** solo la cabeza (incluir los hombros — es lo que da estabilidad y
    robustez a oclusión).
- **Una caja por persona.**

## Por qué cabeza+hombros (no silueta, no cabeza sola)

- **Invariante a la posición en el frame:** la "T" se ve casi igual en el
  centro que en el borde; la silueta completa se deforma (alargada en los
  bordes por perspectiva). El modelo generaliza mejor.
- **Excluye la sombra en el piso**, que de noche/con sol lateral mete ruido si
  se incluye en el bbox.
- **Robusto a oclusión de la cabeza** (gorros, pelo oscuro sobre fondo claro):
  los hombros casi nunca se ocluyen desde arriba.

## Sí etiquetar

- Personas **caminando** (caso operativo principal).
- Personas **estáticas** — sentadas, paradas quietas, en fila. **IMPORTANTE:**
  esto cierra el gap del v1, que no las detecta. Etiquetalas siempre.
- Personas con la **cabeza parcialmente cortada** por el borde del frame, si
  los hombros se ven (el bbox llega hasta donde se ve).
- Cada persona de un **grupo** apretado, por separado (aunque se solapen un
  poco los bboxes).

## NO etiquetar

- **Reflejos** en vidrios/espejos/vitrinas (no son personas reales).
- **Maniquíes**.
- **Personas afuera, a través de la vidriera** (ej. site_54_21 tiene mucha
  gente en la vereda — no entran al conteo del local).
- Una persona de la que **solo se ve un fragmento** (un brazo, un pie) sin
  cabeza ni hombros identificables.

## Casos dudosos

- Si dudás si una mancha es persona o sombra/objeto → **no la etiquetes**
  (preferir precisión en el validation set; un falso positivo en el ground
  truth es peor que un falso negativo).
- Persona agachada / inclinada: el bbox sigue siendo cabeza + hombros tal
  como se ven proyectados desde arriba.

## X-AnyLabeling — setup

1. **Abrir la carpeta** del batch (ej. `training_data/label_val_01/`).
2. Crear la clase `person` (o tipear `person` en el primer bbox; queda fija).
3. **Dibujar rectángulos** (tecla `R` o el botón de rectángulo) — NO usar SAM
   para esto; el bbox de cabeza+hombros se dibuja directo, más rápido y preciso.
4. **Export / formato: YOLO** (`class cx cy w h` normalizado). X-AnyLabeling
   guarda un `.txt` por imagen.
5. Atajos útiles: `D` siguiente imagen, `A` anterior, `Ctrl+S` guardar.

## Después del batch

- Los `.txt` YOLO quedan junto a las imágenes en la carpeta del batch.
- El `manifest.txt` mapea cada copia a su origen en `captures/`.
- El validation set se **reserva** (no se entrena con él) para comparar
  modelos. Los batches de training van en carpetas separadas.
