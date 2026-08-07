# scripts/analysis/ — bancos de análisis reproducibles

Cuatro bancos que explotan material ya existente (traza de operación,
conjunto de validación anotado, y el propio código de seguimiento, conteo y
agrupamiento). Cada uno se ejecuta con **un comando** y escribe su reporte en
Markdown por salida estándar.

Ninguno requiere hardware del dispositivo. El primero necesita el log de
aplicación resguardado; el tercero necesita el modelo ONNX y el conjunto de
validación (ambos fuera de control de versiones por tamaño); el segundo y el
cuarto no necesitan nada externo.

## Qué es medición y qué no

| Banco | Verdad de referencia | Estatus |
|---|---|---|
| `audit_directed_trials.py` | ninguna | **Evidencia documental**, no medición. Reconstruye decisiones del sistema; no hay registro de lo que ocurrió frente a la cámara. No derivar tasas de acá. |
| `simulate_associator.py` | entrada sintética conocida | **Medición sobre la lógica**, con modelo de ruido no calibrado. Caracteriza el algoritmo, no el sistema completo. |
| `eval_detector_valset.py` | anotación humana por imagen | **Medición**. Acotada al dominio del conjunto (cámaras de sucursal) y al modelo ONNX, no al HEF cuantizado. |
| `simulate_wifi_stitching.py` | entrada sintética conocida | **Medición sobre la lógica**, con modelo de emisión no calibrado. Caracteriza el algoritmo ante patrones de rotación conocidos; no predice tasas sobre la población real de dispositivos. |

## 1. Auditoría de la traza de una jornada de ensayos

```bash
python scripts/analysis/audit_directed_trials.py --log debug/app.log --date 2026-06-25
```

Reconstruye visitas a la counting zone: entradas, cruces con lado y balance
neto, veredicto de salida, muertes, adopciones y rechazos por guarda.
Identifica ráfagas candidatas a ensayo dirigido y pares convergentes en
sentidos opuestos.

Opciones útiles: `--csv salida.csv` (volcado por visita), `--converge-px`
(umbral de proximidad para considerar convergente un par), `--line-y` /
`--zone` (geometría de la fecha auditada — el ROI cambió el 2026-06-27, así
que los valores por defecto **sólo** valen para el 2026-06-25).

## 2. Banco de simulación del asociador

```bash
python scripts/analysis/simulate_associator.py --trials 2000 --symmetry-trials 500
```

Prueba de sesgo de dirección (determinista por reflexión exacta + estadística
pareada por McNemar) y, con `--sweep`, la superficie de decisión sobre
separación, velocidad y dirección relativa.

El modelo de ruido de detección está documentado en el encabezado del script
junto con su falta de calibración empírica. Leerlo antes de citar cualquier
número.

## 3. Caracterización del detector

```bash
python scripts/analysis/eval_detector_valset.py --bootstrap 2000
```

Precisión, exhaustividad y AP@0.5 globales y por sucursal, franja horaria y
origen, con intervalos de confianza por bootstrap sobre imágenes; más la tasa
de falsos positivos sobre las imágenes sin ninguna caja anotada.

**Camino de decodificación.** El script usa `src.vision.detect.detect_persons`,
la misma ruta que el resto del repositorio: letterbox, decodificación del
tensor, NMS, fusión por centroide y supresión por contención. La cantidad de
clases se deriva del tensor, de modo que sirve tanto para el detector afinado
a una clase como para un modelo de 80. No hay un decodificador propio del
script que mantener sincronizado.

## 4. Banco de agrupamiento de identidad inalámbrica

```bash
python scripts/analysis/simulate_wifi_stitching.py
```

Alimenta `DedupEngine` con secuencias de emisión sintéticas donde cada emisión
tiene asignado su dispositivo verdadero. Los escenarios ejercitan las cuatro
reglas del esquema: rotación con continuidad de número de secuencia, rotación
con reinicio y huella estable, ambos protocolos simultáneos, y un solo
protocolo.

Reporta falsos agrupamientos (grupos impuros y pares indebidamente
unificados), separaciones incorrectas (dispositivos repartidos y grupos por
dispositivo) y el cociente grupos/dispositivos, que es el sesgo sobre el
indicador de tráfico exterior. Todo con intervalos bootstrap sobre
repeticiones, barrido sobre la cantidad de dispositivos simultáneos.

Opciones: `--devices` (densidades del barrido), `--repetitions`,
`--session-seconds` (por defecto 900, la ventana de publicación del sistema),
`--rotation-seconds`, `--rssi-min/--rssi-max/--rssi-sigma`,
`--emissions-per-min`.

El modelo de emisión y su falta de calibración empírica están documentados en
el encabezado del script. Leerlo antes de citar cualquier número.
