# Resultados de benchmark — People Counter Edge System

## Resumen

| | |
|---|---|
| Casos de prueba | 19 — **17 cumplen su criterio**, 2 con hallazgo documentado (TC-03, TC-06) |
| Requisitos no funcionales | todos los umbrales cumplidos (§3) |
| Estructura de cada registro | procedimiento · criterio de aceptación · resultado con sus datos · procedencia de la evidencia |
| Evidencia primaria | `validation/` — guiones reproducibles y salidas crudas, con [índice](../validation/README.md) |

Este documento es el **registro de las mediciones**. La interpretación, el
encuadre metodológico y la discusión de los resultados están en la memoria del
trabajo final.

## 1. Introducción

Este documento consolida la validación del prototipo de conteo de personas: el
nodo de borde (Raspberry Pi 5 + acelerador Hailo-8L, par estéreo IMX708 con
modelo fisheye Kannala-Brandt, captura pasiva WiFi/BLE) y el backend en AWS
(IoT Core → Lambda → RDS Postgres → Grafana). La validación combina dos planos:
**caracterización de banco** (requisitos no funcionales medidos sobre la unidad
de desarrollo, en laboratorio) y **validación dirigida en las instalaciones de
la organización** (montaje cenital reproduciendo la geometría de instalación,
con cruces controlados y tráfico orgánico operador-confirmado). Esta validación
dirigida tiene carácter **indicativo** —una unidad, muestras del orden de la
decena por caso, un único operador—: confirma el funcionamiento en la
configuración de montaje cenital, pero la medición de exactitud estadísticamente
robusta a escala corresponde a la etapa de **piloto** (futura, sujeta a la
decisión de la organización y desarrollada en el plan de implementación). Los
valores provienen de mediciones instrumentadas reproducibles; cuando un criterio
no pudo verificarse se indica explícitamente.

## 2. Casos de prueba (TC-01 … TC-19)

| Código | Descripción | Muestra | Resultado observado | Veredicto |
|---|---|---|---|---|
| TC-01 | Conteo de ingreso individual | 10 cruces | 10/10 (dirigido controlado) | ✅ PASS |
| TC-02 | Conteo de egreso individual | 10 cruces | 10/10 (dirigido controlado) | ✅ PASS |
| TC-03 | Cruces simultáneos en direcciones opuestas | 2 corridas × 10 cruces (5 pares simultáneos) | 7/10 y 8/10 (umbral ≥ 9/10); **sin fusión de identidades** en ninguna (7/7 y 8/8 trayectorias) | ⚠️ Limitación documentada |
| TC-04 | Ráfaga en el mismo sentido | 2 ráfagas (7 egresos + 5 ingresos) en tráfico real | 7/7 y 5/5 — 12 trayectorias distintas, sin omisión, doble conteo ni fusión; cruces consecutivos a 226 y 184 ms | ✅ PASS |
| TC-05 | Robustez a apariencia (gorra/capucha) | 10 cruces (5 idas y vueltas) | 10/10 — serie completa y alternada; confianza del detector 0,593–0,769 frente a 0,819–0,902 sin accesorio | ✅ PASS |
| TC-06 | Rechazo de objetos por debajo del umbral de altura | 8 pasadas gateando | 1/8 contada (criterio: 0/8). El filtro rechazó las 3 pasadas que midieron bajo el umbral (0,96–0,97 m < 1,00 m); la cuarta lo superó, de modo que no correspondía rechazarla — ver L5 | ⚠️ Cumplimiento parcial |
| TC-07 | Hesitación: entrada a la zona sin cruzar la línea | 8 aproximaciones | 0 eventos; contadores del dispositivo sin variación en tres muestras sucesivas | ✅ PASS |
| TC-08 | Estimación de estatura (±10 cm) | 15 mediciones · 2 sujetos (1,68 y 1,82 m) | 15/15 dentro de tolerancia; error absoluto medio 2,8 cm, máximo 6 cm | ✅ PASS |
| TC-09 | Stitching WiFi por continuidad de identidad | 6 direcciones de 1 dispositivo (prueba de componente) | 6 MACs randomizadas → 1 `group_id`; segundo device → grupo aparte | ✅ PASS |
| TC-10 | Stitching entre protocolos, WiFi y BLE | par WiFi+BLE + control negativo (prueba de componente) | mismo device (ΔRSSI 2 dBm, <2 s) → mismo grupo; control ΔRSSI 38 dBm → grupo aparte | ✅ PASS |
| TC-11 | Tasa de conversión de extremo a extremo | 56 casos (8 sucursales × 7 días) | 56/56 — visitantes, ventas y tasa coinciden con el cálculo directo sobre las tablas base | ✅ PASS |
| TC-12 | Idempotencia de la ingesta en la nube | 20 duplicados × 2 tablas | 40/40 descartados por la restricción de unicidad; 0 filas insertadas | ✅ PASS |
| TC-13 | Consulta de agregados: control de acceso y validación | 4 invocaciones | sin credenciales → 403; falta `from` → 400; rango de 20 d con agrupamiento de 15 min → 400 (RFC 7807); válida → 200 | ✅ PASS |
| TC-14 | Privacidad por diseño | 500 muestras en la nube + auditoría del disco del dispositivo | 0 direcciones MAC en claro; las 3 columnas señaladas por el barrido examinadas y descartadas; 0 imágenes escritas por el pipeline | ✅ PASS |
| TC-15 | Latencia de extremo a extremo (p95 ≤ 5 s) | 4 728 invocaciones (configuración vigente) | p95 0,33 s conteo · 0,43 s telemetría · 0,44 s inalámbrico. La configuración previa al 8-06 no cumplía (p95 6,5–7,2 s) | ✅ PASS |
| TC-16 | Resiliencia ante corte breve de conectividad | 30 eventos (prueba de componente) | encolados offline → drenados íntegros, 0 pérdida / 0 duplicado | ✅ PASS |
| TC-17 | Resiliencia ante corte prolongado | 1 302 eventos = volumen de 72 h (prueba de componente) | persistidos y drenados sin pérdida ni duplicado, muy por debajo del tope de 50 000 mensajes; el control con tope reducido a 1 000 confirma que el acotamiento actúa | ✅ PASS |
| TC-18 | Reinicio tras corte de energía (< 90 s) | 1 corte físico real | frames fluyendo en **+46 s**; `integrity_check` SQLite OK; fs sano, throttle 0x0 | ✅ PASS |
| TC-19 | Disponibilidad del stack cloud (24 h ≥ 99 %) | 24 h (21-06 16:20 → 22-06 16:20, −03) | Medido sobre los servicios, no sobre el device: **RDS 288/288 intervalos (100 %)**, **Fargate `LiveTaskCount` ≥ 1 en 288/288 (100 %)**, **IoT Core 0 fallos del broker** y publicaciones aceptadas en 287/288 (99,7 %). Alarmas: ninguna de las diez alcanzó su condición. Tableros: destino sano en 288/288, 842 peticiones, 0 respuestas 5XX | ✅ PASS |

**TC-01 y TC-02 — conteo de ingreso y de egreso.**

**Procedimiento.** Diez idas y vueltas del operador (estatura real 1,68 m), sin
otras personas en la escena, partiendo del lado interior y regresando a él. Los
veinte cruces resultantes se evalúan por dirección: los diez ingresos
corresponden a TC-01 y los diez egresos a TC-02. No son, por lo tanto, muestras
independientes ni de sujetos distintos: son **n = 1 sujeto**. La ventana va del
instante de referencia (`2026-06-25T11:11:27Z`) al aviso de fin del operador
(`11:14:15Z`) y contiene exactamente veinte eventos, diez de cada dirección, sin
ninguno fuera de ella. Duración total: 135 s.

**Criterio de aceptación.** TC-01: evento de ingreso con dirección **y altura**
correctas en al menos 9 de los 10 cruces. TC-02: evento de egreso con dirección
correcta en al menos 9 de los 10.

**Resultado.** Diez de diez en ambos casos.

**TC-01 — los diez ingresos** (hora local del dispositivo, UTC−3):

| # | Hora | Confianza | Estatura estimada |
|---:|---|---:|---:|
| 1 | 08:12:03 | 0,877 | 1,67 |
| 2 | 08:12:17 | 0,877 | 1,70 |
| 3 | 08:12:30 | 0,877 | 1,70 |
| 4 | 08:12:43 | 0,902 | 1,70 |
| 5 | 08:12:57 | 0,877 | 1,67 |
| 6 | 08:13:11 | 0,884 | 1,70 |
| 7 | 08:13:25 | 0,866 | 1,68 |
| 8 | 08:13:40 | 0,877 | 1,70 |
| 9 | 08:13:56 | 0,895 | 1,70 |
| 10 | 08:14:11 | 0,887 | 1,70 |

Confianza 0,866–0,902 (media 0,882). Estatura media 1,692 m contra 1,68 m
reales, con un **error máximo de 2 cm**: la dirección y la altura que exige el
criterio quedan verificadas sobre la misma tabla.

**TC-02 — los diez egresos**:

| # | Hora | Confianza | Estatura estimada |
|---:|---|---:|---:|
| 1 | 08:11:56 | 0,852 | 1,68 |
| 2 | 08:12:10 | 0,830 | 1,70 |
| 3 | 08:12:23 | 0,873 | 1,71 |
| 4 | 08:12:35 | 0,848 | 1,71 |
| 5 | 08:12:49 | 0,877 | 1,71 |
| 6 | 08:13:04 | 0,841 | 1,71 |
| 7 | 08:13:18 | 0,848 | 1,70 |
| 8 | 08:13:32 | 0,834 | 1,71 |
| 9 | 08:13:47 | 0,819 | 1,71 |
| 10 | 08:14:03 | 0,862 | 1,71 |

Confianza 0,819–0,877 (media 0,848). Estatura media 1,705 m, error máximo 3 cm.

**Por qué el resultado no depende de la instrucción dada.** Las tablas prueban
que se contaron diez ingresos y diez egresos, pero no, por sí solas, que no se
haya perdido ningún cruce. Eso lo prueba la estructura de la serie: los veinte
eventos **alternan egreso e ingreso sin una sola excepción**
(`O I O I O I O I O I O I O I O I O I O I`), de modo que una omisión aislada
habría dejado dos eventos consecutivos de la misma dirección; y los intervalos
van de **6 a 9 s** (media 7,1 s), de modo que una ida y vuelta entera no
registrada habría abierto un hueco de unos 14 s. Ni lo uno ni lo otro aparece.

**Procedencia de la evidencia.** Registro de eventos persistidos en la base de
datos.

**TC-03 — cruces simultáneos en direcciones opuestas.**

**Procedimiento.** Dos personas cruzan el umbral **al mismo tiempo y en sentidos
opuestos**: mientras una entra, la otra sale, de modo que sus trayectorias se
cruzan dentro de la imagen. Cinco pares por corrida, es decir diez cruces
esperados, cinco de cada sentido. Se ejecutaron **dos corridas**, ambas con los
sujetos por **carriles cruzados** —uno en cada sentido—; la segunda repitió el
ejercicio con **algo más de separación lateral entre los carriles**. Sujetos: el
operador (1,68 m) y el segundo sujeto (1,82 m). Instantes de referencia
`2026-06-25T13:15:33Z` y `2026-06-25T13:19:22Z`.

*La descripción del recorrido procede del **operador que ejecutó el ensayo**. El
registro de eventos y la telemetría, que son la evidencia del caso, no permiten
establecer por dónde caminó cada persona: acreditan qué se contó y cuándo, no la
disposición espacial.*

**Criterio de aceptación.** Al menos 9 de los 10 cruces contados con su sentido
correcto, y sin fusión de identidades entre los dos sujetos.

**Resultado.** El umbral de conteo **no se alcanza en ninguna de las dos
corridas**; la condición de identidades se cumple en ambas.

**Primera corrida** (hora local del dispositivo, UTC−3):

| # | Hora | Sentido | Confianza | Estatura |
|---:|---|---|---:|---:|
| 1 | 10:15:47 | egreso | 0,812 | 1,85 |
| 2 | 10:15:54 | egreso | 0,765 | 1,71 |
| 3 | 10:16:00 | egreso | 0,740 | 1,85 |
| 4 | 10:16:06 | ingreso | 0,417 | — |
| 5 | 10:16:07 | egreso | 0,801 | 1,69 |
| 6 | 10:16:14 | egreso | 0,740 | 1,79 |
| 7 | 10:16:14 | ingreso | 0,834 | 1,68 |

Siete eventos sobre diez esperados: 2 ingresos y 5 egresos. Faltan **tres
ingresos**. Los siete corresponden a **siete trayectorias distintas**.

**Segunda corrida** — con mayor separación lateral entre carriles:

| # | Hora | Sentido | Confianza | Estatura |
|---:|---|---|---:|---:|
| 1 | 10:19:32 | ingreso | 0,586 | 1,68 |
| 2 | 10:19:39 | ingreso | 0,446 | — |
| 3 | 10:19:39 | egreso | 0,672 | 1,70 |
| 4 | 10:19:47 | ingreso | 0,557 | 1,68 |
| 5 | 10:19:53 | ingreso | 0,546 | 1,86 |
| 6 | 10:19:54 | egreso | 0,672 | 1,71 |
| 7 | 10:20:02 | egreso | 0,668 | 1,80 |
| 8 | 10:20:02 | ingreso | 0,647 | 1,67 |

Ocho eventos: 5 ingresos y 3 egresos. Faltan **dos egresos**. Ocho eventos,
**ocho trayectorias distintas**.

La segunda corrida cuenta un cruce más que la primera, pero **de ahí no se sigue
que lo haya causado la mayor separación**: un cruce de diferencia sobre diez,
entre dos corridas de cinco pares, cae dentro de lo que puede variar por azar.
Lo único que puede decirse es que la diferencia va en la **dirección** que
predice el banco de simulación —menos pérdida a mayor separación lateral— y
nada más que eso. Ninguna de las dos alcanza el umbral.

La base registra un ingreso más a las 10:20:46, **cuarenta y cuatro segundos
después** de cerrarse el quinto par. Queda fuera de la secuencia —los pares se
suceden cada seis a ocho segundos— y por eso no entra en el resultado.

Los cruces faltantes **ocurrieron físicamente**: los ejecutó el operador, que
confirmó ambas secuencias en el momento. No se trata de pares que no llegaron a
producirse, sino de cruces reales no registrados.

**Sin fusión de identidades.** Cada evento nace de una trayectoria propia:
siete sobre siete en la primera corrida y ocho sobre ocho en la segunda. Ninguna
identidad se fusiona ni se intercambia.

**El modo de falla, legible en la estructura de pares.** En la segunda corrida
los eventos se agrupan solos: `10:19:39` trae un ingreso y un egreso **en el
mismo segundo**, `10:19:53`–`10:19:54` forman otro par, y `10:20:02` un tercero,
también en el mismo segundo. Quedan dos ingresos sueltos, sin su egreso
correspondiente. La primera corrida muestra lo mismo con los sentidos invertidos:
dos pares completos y tres egresos sin su ingreso. El patrón es constante: **de
cada par simultáneo que falla se pierde exactamente uno de los dos cruces, nunca
los dos**. El sistema no deja de ver la escena; deja de registrar uno de los dos
movimientos en disputa.

El sentido que se pierde no es el mismo en las dos corridas —ingresos en la
primera, egresos en la segunda—, de modo que el error del conteo neto tampoco
tiene un signo constante: −3 en una y +2 en la otra. Sobre este registro, la
omisión no muestra un sesgo sistemático de sentido, aunque cada corrida
individual sí quede desbalanceada.

**Qué no puede afirmarse sobre la causa.** El registro disponible no permite
aislarla, y las tres explicaciones candidatas quedan **abiertas**:

| Hipótesis | Qué dice el registro |
|---|---|
| Detector | La confianza media es **0,730** en la primera corrida y **0,599** en la segunda, contra **0,865** en TC-01/TC-02 con el mismo detector y sujeto. Es una caída del mismo orden que la que produce la capucha en TC-05, de modo que no puede descartarse |
| Tasa de cuadros | La telemetría informa 23,9 FPS antes de las corridas y **12,8** y **12,2** en las muestras que las contienen. Son promedios de cinco minutos con la escena mayormente vacía y el regulado de ralentí activo, por lo que no acreditan que los cruces se procesaran a tasa baja —el despertar es inmediato—, pero tampoco permiten excluir la tasa como factor |
| Convergencia del seguidor | La razón de fragmentación de identidades sube de 1,06 a **1,33** en la muestra que contiene la primera corrida: doce identificadores de trayectoria para nueve conteos. Es consistente con trayectorias que se crean y no llegan a registrar su cruce, pero es un indicador agregado del día y no una medición del episodio |

**Caracterización del algoritmo por simulación.** El repositorio incluye un
banco —`scripts/analysis/simulate_associator.py`— que alimenta la lógica de
asociación y conteo con trayectorias sintéticas de **verdad conocida** y mide
qué fracción de cruces no llega a contabilizarse. No sustituye al ensayo de
campo ni revisa su veredicto: **caracteriza el algoritmo, no el sistema
completo**, y su modelo de ruido —probabilidad de pérdida de detección,
dispersión de posición y de profundidad— **no está calibrado empíricamente**,
porque no se conservó video del prototipo con el cual ajustarlo. Así está
declarado en el encabezado del guion y en `scripts/analysis/README.md`. Lo que
aporta es el **régimen** en que aparece el modo de falla, que el ensayo de campo
no puede barrer.

```
py scripts/analysis/simulate_associator.py --trials 2000 --symmetry-trials 500
py scripts/analysis/simulate_associator.py --sweep --sweep-trials 120
```

*Semilla por defecto 20260803. El barrido se corre por separado porque las fases
A1 y A2 consumen el generador de aleatorios: encadenarlos cambia los valores del
barrido para la misma semilla.*

**A1 — simetría por reflexión exacta.** Cada ensayo se corre junto a su reflejo
puntual, con la misma realización de ruido. Sobre **500 pares**, **cero
discrepancias** respecto del espejo.

**A2 — prueba estadística pareada**, a la separación nominal de 60 px y sobre
**2 000 ensayos**: pérdida de ingreso 0/2000 y de egreso 0/2000, diferencia
+0,0000 con intervalo de confianza del 95 % por remuestreo de [+0,0000,
+0,0000] y McNemar exacto bilateral p = 1,0000. Los rechazos por ambigüedad del
asociador suman 15 en total, 0,01 por ensayo, con un máximo de 2.

**B — superficie de decisión**, 120 ensayos por punto. Fracción de cruces no
contabilizados en **sentidos opuestos**, que es la condición de TC-03:

| Separación | v = 6 px/cuadro | v = 12 | v = 24 |
|---:|---:|---:|---:|
| 10 px | 0,100 | 0,046 | 0,017 |
| 20 px | 0,075 | 0,008 | 0,008 |
| 40 px | 0,017 | 0,000 | 0,000 |
| 60 px y más | 0,000 | 0,000 | 0,000 |

**La pérdida aparece por debajo de unos 40 px de separación lateral y desaparece
a partir de 60 px, y empeora cuanto más lento es el cruce** —a 10 px de
separación pasa de 0,017 a 0,100 al bajar la velocidad de 24 a 6 px por
cuadro—, porque a menor velocidad las dos trayectorias permanecen más cuadros
dentro de la zona de confusión.

El barrido incluye un **control en el mismo sentido**, dos personas avanzando a
la par con separación lateral constante, que mide otra cosa y conviene no
confundir:

| Separación | Pérdida (las tres velocidades) |
|---:|---|
| 10 a 100 px | 0,500 |
| 150 px | 0,000 · 0,008 · 0,054 |
| 250 px | 0,000 |

Ese 0,500 sostenido hasta los 100 px no es un fallo del asociador: las dos
personas quedan **fusionadas por el agrupamiento de centroides previo al
seguidor** durante todo el trayecto —el umbral es `detection.cluster_distance_px`
= 150 px— y de cada par se cuenta una sola. Es un **peor caso geométrico**, no
una ráfaga real: en una ráfaga las personas van escalonadas en el sentido de la
marcha, que es la condición que TC-04 mide sobre tráfico real y donde se
contaron las doce trayectorias sin fusión.

**Segunda parametrización — régimen adverso.** A la separación nominal la
pérdida es **cero en ambos sentidos**, de modo que la prueba de simetría queda
sin material: no hay pérdidas que comparar entre un sentido y el otro. Para
poder ponerla a prueba hace falta un régimen donde el algoritmo efectivamente
falle, y por eso se corre una segunda vez en un punto **deliberadamente
adverso** —separación 10 px, probabilidad de pérdida de detección 0,25 y
velocidad 24 px por cuadro—, que **no es el punto de operación** sino el peor
rincón de la superficie de decisión.

```
py scripts/analysis/simulate_associator.py --trials 2000 --symmetry-trials 500 \
    --separation-px 10 --p-miss 0.25 --speed-px-frame 24
```

Ahí sí hay pérdida que medir, y la simetría se sostiene por dos vías
independientes:

| | Resultado |
|---|---|
| A1 — reflexión exacta, 500 pares | **0 discrepancias** respecto del espejo |
| Pérdida de ingreso | 64/2000 = 0,032 (IC 95 % Wilson 0,025–0,041) |
| Pérdida de egreso | 70/2000 = 0,035 (IC 95 % Wilson 0,028–0,044) |
| Pares discordantes | b = 42 (sólo ingreso), c = 48 (sólo egreso) |
| Diferencia (ingreso − egreso) | **−0,0030** (IC 95 % por remuestreo −0,0120 a +0,0060) |
| McNemar exacto bilateral | **p = 0,5984** |
| Rechazos por ambigüedad | 671 en total, 0,34 por ensayo, máximo 5 |

La prueba determinista no encuentra **ninguna** asimetría en 500 reflexiones, y
la estadística acota la diferencia entre sentidos a ±1,2 puntos porcentuales sin
poder distinguirla de cero. Con una pérdida del orden del 3 % en ese régimen,
esos dos resultados juntos son lo que permite afirmar que **el algoritmo no
tiene sesgo de dirección detectable**.

Conviene no confundir las dos corridas: las cifras de arriba pertenecen a un
punto adverso escogido para hacer visible el fenómeno, y no describen el
comportamiento esperado en operación, que es el de las tablas anteriores.

**Qué agrega y qué no.** Sitúa el modo de falla de TC-03 en el régimen de
separación lateral pequeña y velocidad baja, y aporta dos hechos que el ensayo
de campo no pudo establecer: que el asociador no tiene sesgo de dirección
detectable, y que por encima de 60 px de separación la pérdida desaparece en el
modelo. No mide el sistema desplegado, de modo que **el veredicto de TC-03 no se
revisa**: sigue siendo el del ensayo de campo.

**Procedencia.** `validation/tc03_simulacion_result.txt`, salida íntegra de las
dos corridas.

**Instrumentación posterior.** El indicador `ambiguous_reject_count` **no
existía durante estas corridas** —su primer valor no nulo es de las 12:01, casi
dos horas después—, de modo que no forma parte de la evidencia. Se incorporó
después y hoy recorre la cadena desde el dispositivo hasta el tablero.

**Procedencia de la evidencia.** Registro de eventos persistidos en la base de
datos y telemetría del dispositivo del mismo período, más la salida del banco
de simulación en `validation/tc03_simulacion_result.txt`. La traza por repetición de
toda la jornada está en `validation/tc_trace.csv`, con su lectura en
`validation/tc_audit.md`; el banco que fija los criterios es
`validation/tc_controlled.py`.

**TC-04 — ráfaga en el mismo sentido.**

**Procedimiento.** El caso se resolvió sobre **tráfico real**, aprovechando que el dispositivo ya estaba instalado y en operación y que
el operador se encontraba controlándolo en sitio. Montar la versión coreografiada
habría exigido reunir a un grupo de personas para que cruzara a demanda; en
cambio, el propio tránsito de personas produjo dos ráfagas del mismo sentido, que
el operador identificó y confirmó en el momento. La condición resultante es **más
exigente** que la planificada: nadie ajustó el ritmo ni la separación entre
personas.

Se planificaron dos ráfagas de cinco personas. Por tratarse de tráfico no
coreografiado, la de egreso resultó de **siete** y la de ingreso de **cinco**.
Ambas fueron revisadas y confirmadas por el operador en el momento del ensayo.

**Criterio de aceptación.** Todos los cruces contados, sin omisión, sin doble
conteo y sin fusión de identidades.

**Resultado.** Siete de siete y cinco de cinco.

**Ráfaga de egreso — siete personas en 28 s** (hora local del dispositivo):

| # | Hora | Confianza | Estatura estimada |
|---:|---|---:|---:|
| 1 | 15:51:44,199 | 0,801 | 1,61 |
| 2 | 15:51:56,710 | 0,862 | 1,50 |
| 3 | 15:51:59,501 | 0,855 | 1,42 |
| 4 | 15:52:00,772 | 0,841 | 1,70 |
| 5 | 15:52:00,998 | 0,898 | 1,51 |
| 6 | 15:52:02,379 | 0,611 | 1,50 |
| 7 | 15:52:11,996 | 0,816 | 1,71 |

Siete cruces, **siete trayectorias distintas**, sin doble conteo ni fusión de
identidades. Dos de ellos —los track 32 y 35— quedaron separados por **226
milésimas de segundo**, que es el caso difícil que el criterio busca: personas
saliendo prácticamente juntas. Sobre esta ráfaga la verificación es **exacta**:
se preguntó al operador si en ese grupo habían salido siete personas «ni más ni
menos» y lo confirmó, de modo que quedan descartados tanto los conteos espurios
como las omisiones.

**Ráfaga de ingreso — cinco personas en 9 s**:

| # | Hora | Confianza | Estatura estimada |
|---:|---|---:|---:|
| 1 | 15:57:58,533 | 0,798 | 1,61 |
| 2 | 15:58:01,093 | 0,826 | 1,73 |
| 3 | 15:58:02,513 | 0,887 | 1,52 |
| 4 | 15:58:07,030 | 0,895 | 1,63 |
| 5 | 15:58:07,214 | 0,887 | 1,51 |

Cinco cruces, cinco trayectorias distintas, con dos separados por **184
milésimas**. El operador confirmó estos cruces como correctos, lo que descarta
conteos espurios; a diferencia de la ráfaga anterior, no se formuló sobre ella
una pregunta específica por la cantidad exacta de personas.

Las estaturas estimadas del grupo abarcan de 1,42 a 1,73 m y resultaron
coherentes con la composición del grupo observada por el operador. Deben leerse,
sin embargo, con dos reservas. La primera es la tolerancia declarada de la
estimación, de **±10 cm**. La segunda es que estas ráfagas son **anteriores a la
mitigación de la compresión de estatura** descrita como L1 en §4, aplicada al día
siguiente: los valores de este ensayo arrastran esa compresión y los extremos
bajos del rango deben tomarse con esa reserva. No son, por lo tanto, mediciones
de estatura verificadas —esa verificación corresponde a TC-08, posterior a la
mitigación—, sino la constatación de que el sistema resolvió individualmente a un
grupo heterogéneo sin colapsar su dispersión en un valor único.

**Procedencia de la evidencia.** Registro de eventos persistidos en la base de
datos y confirmación del operador registrada en el momento del ensayo.

**TC-05 — robustez a la variación de apariencia.**

**Procedimiento.** El mismo operador de TC-01 y TC-02 (estatura real 1,68 m)
repitió cinco idas y vueltas **con capucha puesta**, sin otras personas en la
escena y con el mismo recorrido que en aquellos casos: parte del lado interior,
sale, y vuelve a entrar.

**Criterio de aceptación.** Cruce detectado y contado, sin omisión ni doble
conteo, en al menos 9 de los 10 cruces.

**Resultado.** Diez de diez (hora local del dispositivo, UTC−3):

| # | Hora | Dir. | Confianza | Estatura estimada |
|---:|---|---|---:|---:|
| 1 | 08:23:58 | out | 0,751 | 1,70 |
| 2 | 08:24:21 | in | 0,647 | 1,70 |
| 3 | 08:24:27 | out | 0,769 | 1,71 |
| 4 | 08:24:34 | in | 0,722 | 1,70 |
| 5 | 08:24:40 | out | 0,661 | 1,71 |
| 6 | 08:24:47 | in | 0,715 | 1,70 |
| 7 | 08:24:53 | out | 0,593 | 1,71 |
| 8 | 08:25:00 | in | 0,704 | 1,65 |
| 9 | 08:25:06 | out | 0,751 | 1,71 |
| 10 | 08:25:13 | in | 0,596 | 1,70 |

**Por qué la serie está completa.** Los diez eventos alternan egreso e ingreso
sin una sola excepción (`O I O I O I O I O I`), abren con un egreso y cierran con
un ingreso: cinco recorridos de ida y vuelta cerrados, con el operador partiendo
del lado interior y regresando a él. Una omisión habría roto esa alternancia
—dejando dos eventos consecutivos en la misma dirección y al operador del mismo
lado dos veces seguidas— y no ocurre en ningún punto. La cadencia posterior al
primer cruce es además uniforme, de 5,9 a 7,1 s, sin huecos que sugieran un
recorrido no registrado. El ensayo corrió a tasa plena de cuadros —24,7 y 27,5
fotogramas por segundo en las muestras del período—, de modo que tampoco hubo
reducción del régimen de procesamiento.

Ocho segundos antes del primer cruce, a las 08:23:50, la base registra un
ingreso: es el operador entrando a la zona para posicionarse. No pertenece a
las cinco idas y vueltas y queda fuera de la ventana del ensayo.

**El efecto de la capucha, medido.** La comparación con TC-01 y TC-02 es
controlada: mismo sujeto, mismo recorrido, misma geometría, misma jornada, y como
única variable el accesorio que altera la silueta superior de la cabeza.

| | Confianza del detector |
|---|---|
| Sin accesorio (TC-01 / TC-02) | 0,819 – 0,902 · media **0,865** |
| Con capucha (TC-05) | 0,593 – 0,769 · media **0,691** |

La confianza cae unos 0,17 puntos y **ninguna medición con capucha alcanza el
valor mínimo observado sin ella**: los dos rangos no se solapan. El conteo, en
cambio, no se degrada —los diez cruces se registran—, de modo que el caso cumple
su criterio con la evidencia del detector claramente reducida pero todavía
suficiente.

La estimación de estatura tampoco se degrada: 1,65–1,71 m con media 1,699 contra
1,68 m reales, error máximo de 3 cm, en línea con los valores sin accesorio.

**Procedencia de la evidencia.** Registro de eventos persistidos en la base de
datos.

**TC-06 — rechazo de objetos por debajo del umbral de altura.**

**Procedimiento.** El operador atraviesa el umbral **gateando**, para presentar
al sistema un objeto móvil de altura muy inferior a la de una persona de pie: es
el sustituto practicable de un animal o un carro, que no pueden hacerse cruzar a
voluntad. Ocho pasadas, cuatro en cada sentido, alternadas, atravesando de un
lado al otro en cada una. Ventana `18:52:44`–`18:53:59`, hora local del
dispositivo. El equipo corrió con el registro de diagnóstico del seguidor
habilitado, que es lo que permite observar las **decisiones** del filtro y no
sólo sus consecuencias.

**Criterio de aceptación.** Cero conteos sobre las ocho pasadas, con al menos un
rechazo atribuible al filtro de altura, cuyo umbral es 1,00 m. Textualmente, en
el banco de pruebas: «0/8 conteos (detección rechazada por gate de altura)».

**Resultado.** **El criterio estricto no se alcanza**: una de las ocho pasadas
quedó contada. El reparto de las ocho:

| Desenlace | Pasadas |
|---|---:|
| Sin trayectoria confirmada ni evento | 4 |
| Detectadas y **rechazadas por el filtro de altura** | 3 |
| Detectadas y contadas, con altura por encima del umbral | 1 |

Detalle de las cuatro que llegaron a producir una decisión:

| Hora | Altura medida | Desenlace |
|---|---|---|
| 18:52:44 | 0,96 m | rechazada por el filtro |
| 18:53:09 | 0,96 m | rechazada por el filtro |
| 18:53:34 | ≥ 1,00 m (inferida) | **contada** — no publicada por la compuerta de confianza (0,453) |
| 18:53:59 | 0,97 m | rechazada por el filtro |

La base de datos contiene exactamente un evento en la ventana, el de las
18:53:34. Los posteriores a las 18:55 corresponden a personas que circularon una
vez terminado el ensayo, con confianzas de 0,84 a 0,88 y estaturas de 1,67 a
1,70 m, netamente separadas de las del ensayo.

**El filtro actuó como está diseñado en las cuatro pasadas.** Tres midieron
0,96, 0,96 y 0,97 m —coherentes entre sí y con la postura del sujeto, no
rechazos por azar ni por ruido— y quedaron por debajo del umbral de 1,00 m: el
filtro las rechazó. La cuarta **también tenía altura medida, y por encima del
umbral**, de modo que no había motivo para rechazarla.

**Por qué la cuarta se contó, y por qué el registro parecía decir otra cosa.**
En la base ese evento figura con altura **nula**, lo que sugiere que no se midió.
No es así: la altura nula es la que se **suprime en el reporte**, no la que se
mide. Cuando la mediana de confianza del detector cae por debajo de
`height_confidence_gate` —0,5 por defecto— la altura no se publica, porque con
recuadros marginales la estatura derivada no es confiable; el conteo sí se
mantiene, que es el dato primario. Esa pasada tuvo mediana de confianza 0,453 y
por eso su altura viajó nula.

La altura **cruda** existía, y puede acotarse por lo que hicieron los dos guards
del counter. El de altura mínima rechaza cuando la medición existe y cae bajo
1,00 m: no actuó. El de baja confianza rechaza cuando **no** hay medición y la
confianza cae bajo 0,60: con 0,453 el segundo requisito se cumplía, así que si
no actuó fue porque el primero no: había medición. Las dos cosas juntas sitúan
la altura mediana de esa pasada **en 1,00 m o algo por encima**.

*Es una inferencia, no una lectura: el valor crudo no se conserva. El registro
de diagnóstico sólo escribe la altura cuando un guard la rechaza o cuando el
evento la publica, y en esa pasada no ocurrió ninguna de las dos.*

**Qué significa para el criterio.** El caso no lo alcanza, pero no porque el
mecanismo fallara: el sujeto gateando alcanzó en esa pasada una altura mediana
por encima del metro, y un filtro geométrico con umbral en 1,00 m no tenía por
qué descartarla. Es la limitación **L5** de §4: cerca del umbral, la dispersión
del estimador es del mismo orden que el margen.

**Alcance de la evidencia.** El filtro fue ejercitado en **cuatro de las ocho**
pasadas. En las otras cuatro nunca llegó a existir una trayectoria que evaluar,
de modo que no informan sobre el filtro: contribuyen al resultado observable
—ninguna dejó evento— pero no a la validación del mecanismo. Con el registro
disponible tampoco puede distinguirse si en ellas el detector no llegó a
disparar, o si disparó en pocos cuadros y la trayectoria murió antes de
confirmarse, porque esa muerte no deja constancia. El documento no atribuye la
ausencia a ninguna de las dos causas.

**Procedencia de la evidencia.** Registro de eventos persistidos en la base de
datos y registro de diagnóstico del seguidor en el dispositivo
(`exit_short_height_skipped`), del mismo período; reconstruidos en
`validation/tc_audit.md` y `validation/tc_trace.csv`. El
criterio citado es el de `validation/tc_controlled.py`.

**TC-07 — hesitación sin cruce de la línea.**

**Procedimiento.** Ocho aproximaciones del operador a la zona de conteo
**entrando en ella pero sin cruzar la línea virtual**, con retroceso inmediato.
El diseño es deliberado: cruzar hacia adentro y volver a cruzar hacia afuera
pondría a prueba la cancelación del par de cruces, mientras que entrar sin
cruzar aísla el modo de falla que interesa —que la extrapolación del seguidor,
cuando pierde al sujeto dentro de la zona, no registre un cruce que no ocurrió—.
Es el único procedimiento que garantiza que la línea no se atraviesa. Ventana:
del instante de referencia (`2026-06-25T11:28:12Z`) al aviso de fin del operador
(`11:29:13Z`).

**Criterio de aceptación.** Ninguna de las ocho aproximaciones genera evento de
conteo.

**Resultado.** Cero eventos. Los contadores acumulados del propio dispositivo
permanecen en `in = 24, out = 25` antes, durante y después de la ventana,
verificados en tres muestras sucesivas (08:27:53, 08:28:54 y 08:29:54), y la
base de datos no registra ningún evento en el período. El pipeline operó a tasa
plena —18,2 a 27,3 fotogramas por segundo—, de modo que la extrapolación del
seguidor dispuso de material suficiente para producir un cruce espurio si el
mecanismo hubiera fallado.

**Alcance de la evidencia.** El registro acredita la **ausencia de conteos**. La
detección de las aproximaciones en sí no quedó instrumentada en esa ventana: el
registro de diagnóstico del seguidor se habilitó minutos después. La premisa de
que el sujeto fue detectado se apoya, por lo tanto, en que el mismo detector
resolvió sin fallas los treinta y dos cruces del resto de la jornada, con el
mismo sujeto, la misma geometría y la misma tasa de cuadros.

**Procedencia de la evidencia.** Contadores acumulados del dispositivo y
registro de eventos persistidos en la base de datos.

**TC-08 — estimación de estatura.**

**Procedimiento.** Sujetos de estatura conocida, medida con cinta, cruzan bajo
la cámara en su geometría cenital definitiva; se lee la estatura estimada que el
dispositivo publica con cada evento y se contrasta con el valor real. Se
registraron **quince mediciones sobre dos sujetos**, en tres tandas (hora local
del dispositivo, UTC−3):

| Tanda | Ventana | Sujeto | Mediciones |
|---:|---|---|---:|
| 1 | 08:11:56 – 08:12:17 | operador, 1,68 m | 4 |
| 2 | 09:33:03 – 09:33:31 | segundo sujeto, 1,82 m | 7 |
| 3 | 09:41:52 – 09:42:13 | operador, 1,68 m | 4 |

La primera tanda **reutiliza los cuatro primeros cruces de TC-01 y TC-02**: son
mediciones válidas del mismo sujeto bajo la misma geometría y no había motivo
para repetirlas. Entre esa tanda y las otras dos —a las 09:32:44, diecinueve
segundos antes de la segunda— se aplicó la mitigación de la limitación L1, la
compresión de la estatura estimada en montaje bajo que se describe en §4.

**Criterio de aceptación.** Error absoluto no mayor que 10 cm en cada medición,
sobre al menos dos sujetos de estatura distinta.

**Resultado.** Quince de quince dentro de tolerancia, con **error absoluto medio
de 2,8 cm** y máximo de 6 cm.

| # | Hora | Sujeto | Real | Estimada | Error |
|---:|---|---|---:|---:|---:|
| 1 | 08:11:56 | operador | 1,68 | 1,68 | 0 cm |
| 2 | 08:12:03 | operador | 1,68 | 1,67 | −1 cm |
| 3 | 08:12:10 | operador | 1,68 | 1,70 | +2 cm |
| 4 | 08:12:17 | operador | 1,68 | 1,70 | +2 cm |
| 5 | 09:33:03 | segundo | 1,82 | 1,76 | −6 cm |
| 6 | 09:33:06 | segundo | 1,82 | 1,78 | −4 cm |
| 7 | 09:33:11 | segundo | 1,82 | 1,78 | −4 cm |
| 8 | 09:33:15 | segundo | 1,82 | 1,78 | −4 cm |
| 9 | 09:33:20 | segundo | 1,82 | 1,76 | −6 cm |
| 10 | 09:33:26 | segundo | 1,82 | 1,85 | +3 cm |
| 11 | 09:33:31 | segundo | 1,82 | 1,78 | −4 cm |
| 12 | 09:41:52 | operador | 1,68 | 1,70 | +2 cm |
| 13 | 09:41:59 | operador | 1,68 | 1,69 | +1 cm |
| 14 | 09:42:06 | operador | 1,68 | 1,70 | +2 cm |
| 15 | 09:42:13 | operador | 1,68 | 1,69 | +1 cm |

El sesgo no es simétrico entre sujetos: el de 1,68 m se sobreestima levemente
(+1 cm de media) y el de 1,82 m se subestima (−3,6 cm de media). Es el residuo
de la compresión de L1, que actúa sobre el extremo alto del rango y que la
mitigación reduce sin eliminar del todo.

**Por qué la tanda anterior al ajuste conserva validez.** El operador quedó
medido a ambos lados del ajuste, con el mismo sujeto, la misma geometría y la
misma jornada, de modo que su efecto no hay que suponerlo: está medido.

| Tanda del operador | Media | Error |
|---|---:|---:|
| Antes del ajuste (tanda 1) | 1,688 m | +0,8 cm |
| Después del ajuste (tanda 3) | 1,695 m | +1,5 cm |

La diferencia es de **7 mm**, un orden de magnitud por debajo de la tolerancia
del caso. Sobre el sujeto de 1,82 m, en cambio, el mismo ajuste movió el error
medio de −9 cm a −4 cm según la comparación registrada esa mañana. Ambos hechos
apuntan a lo mismo: la corrección operaba en el extremo alto del rango y dejaba
prácticamente intacta la lectura del sujeto de 1,68 m, que es la razón por la
que las mediciones previas siguen siendo comparables con las posteriores.

**Procedencia de la evidencia.** Registro de eventos persistidos en la base de
datos. Las quince mediciones, en `validation/height_mae.csv`, recogidas con
`validation/height_mae.py`.

**TC-09 — stitching WiFi por continuidad de identidad.**

**Procedimiento.** Prueba de componente sobre el motor de deduplicación de
producción: se le inyecta un patrón de detecciones de verdad conocida —un
dispositivo A que rota **seis direcciones MAC aleatorizadas**, y un dispositivo B
distinto como control negativo— y se observa a qué identidad de grupo las
asigna. Corre sobre una base temporal, sin tocar la del dispositivo.

Entrada **sintética**: sobre tráfico capturado no es observable cuántos
dispositivos físicos había, porque la aleatorización de direcciones lo impide.

**Criterio de aceptación.** Las seis direcciones del dispositivo A colapsan a un
único identificador de grupo, y el dispositivo B queda en un grupo distinto.

**Resultado.** Cumple. Seis direcciones → un grupo, que es la razón de
agrupamiento ideal. El dispositivo B queda separado.

**Procedencia de la evidencia.** `validation/tc09_10_stitching.py` y su
salida `tc09_10_result.txt`.

**TC-10 — stitching entre protocolos, WiFi y BLE.**

**Procedimiento.** Mismo banco que TC-09. Se inyecta una detección WiFi y una
BLE **del mismo dispositivo**, separadas por menos que la ventana de correlación
de 2 s y con potencia recibida casi idéntica —2 dBm de diferencia, dentro del
margen de 5 dBm—. Como control negativo, una detección BLE de otro dispositivo
con 38 dBm de diferencia, muy fuera de ese margen.

**Criterio de aceptación.** El par del mismo dispositivo colapsa a un grupo; el
control queda en un grupo aparte.

**Resultado.** Cumple en ambos sentidos: el par se une y el control se separa.
Que el control sea explícito importa, porque una regla de unión demasiado laxa
—que uniera todo— también satisfaría la primera mitad del criterio.

**Procedencia de la evidencia.** `validation/tc09_10_stitching.py` y su
salida `tc09_10_result.txt`.

**TC-11 — tasa de conversión de extremo a extremo.**

**Procedimiento.** Se recorre la cadena completa que consumen los tableros
—eventos de conteo y transacciones de punto de venta, hacia las tablas de
consolidación, hacia las vistas de agregación, hasta la tasa de conversión— y se
contrasta cada magnitud contra un **valor esperado calculado de forma
independiente**: directamente sobre las tablas base, sin tocar consolidaciones ni
vistas, aplicando el mismo criterio de día local. Ventana del 1 al 7 de junio de
2026 sobre las sucursales de demostración: **56 casos** de sucursal por día.

Datos **sintéticos** (sucursales de demostración). Lo verificado es la cadena
de cómputo, no el dato: que consolidación y vistas no se desvíen del cálculo
directo.

**Criterio de aceptación.** Coinciden las tres magnitudes —visitantes, ventas y
tasa— en el 100 % de los casos, con tolerancia de 1×10⁻⁹ por el redondeo del
punto flotante.

**Resultado.** Cincuenta y seis de cincuenta y seis. Muestra de la salida:

| Sucursal | Día | Visitantes (cadena / esperado) | Ventas (cadena / esperado) | Tasa (cadena / esperada) |
|---|---|---|---|---|
| demo-01 | 2026-06-01 | 176 / 176 | 8 / 8 | 0,045455 / 0,045455 |
| demo-02 | 2026-06-06 | 375 / 375 | 14 / 14 | 0,037333 / 0,037333 |
| demo-08 | 2026-06-07 | 100 / 100 | 8 / 8 | 0,080000 / 0,080000 |

**Procedencia de la evidencia.** `validation/tc11_conversion_rate.py` y su
salida `tc11_conversion_result.txt`, que lista los 56 casos.

**TC-12 — idempotencia de la ingesta en la nube.**

**Procedimiento.** Se eligen **veinte filas existentes por tabla**, en orden
determinista, y se reinsertan copiando todas sus columnas salvo dos: la clave
primaria subrogada, que lleva un identificador nuevo —para que el rechazo lo
produzca la restricción de unicidad de negocio y no el choque de la primaria—, y
las columnas generadas, que la base recomputa. Se aplica a las dos tablas que
tienen restricción de unicidad. Todo corre dentro de una transacción que sólo se
confirma si las cuarenta reinserciones fueron descartadas; si alguna entrara, se
revierte, para no dejar duplicados en una base con datos.

**Criterio de aceptación.** Las veinte duplicadas de cada tabla son rechazadas
por la restricción, y ninguna llega a insertarse.

**Resultado.** Cuarenta de cuarenta descartadas.

| Tabla | Restricción de unicidad | Filas antes | Descartadas | Insertadas | Filas después |
|---|---|---:|---:|---:|---:|
| `count_events` | device, instante, trayectoria y sentido | 314 247 | 20/20 | 0 | 314 247 |
| `pos_transactions` | sucursal y número de transacción | 8 933 | 20/20 | 0 | 8 933 |

**Procedencia de la evidencia.** `validation/tc12_idempotency.py` y su
salida `tc12_idempotency_result.txt`.

**TC-13 — control de acceso y validación de parámetros de la interfaz de consulta.**

**Procedimiento.** Cuatro invocaciones contra la interfaz de consulta de
agregados. La del control de acceso se ejecuta como **petición HTTP real** contra
el punto de entrada desplegado, sin credenciales. Las otras tres invocan la
función directamente con el evento que le entregaría la puerta de enlace, que es
la forma de ejercitar la validación de parámetros sin depender de credenciales
firmadas.

**Criterio de aceptación.** Rechazo sin credenciales; error estructurado según
RFC 7807 ante parámetros inválidos; respuesta con agregados ante una petición
bien formada.

**Resultado.** Las cuatro responden lo esperado.

| Invocación | Respuesta | Contenido |
|---|---|---|
| Sin credenciales (HTTP real) | **403** | `{"message":"Forbidden"}` |
| Falta el parámetro `from` | **400** `application/problem+json` | tipo `missing-parameter`, detalle «`'from'` is required» |
| Rango de 20 días con agrupamiento de 15 min | **400** `application/problem+json` | tipo `range-too-large`, detalle «Requested 20.0 days with bucket=15min, max allowed is 7d» |
| Petición bien formada | **200** | agregados, con validador de caché, política de caché y encabezado de paginación |

Los dos errores llegan con `content-type: application/problem+json` y el cuerpo
completo que exige RFC 7807 —tipo, título, estado, detalle, instancia y el
parámetro ofensor—, de modo que el consumidor puede distinguir la causa sin
interpretar texto libre.

**Procedencia de la evidencia.** Pares de entrada y salida en
`validation/`: `validation/tc13_api_checks.py` y su salida
`validation/tc13_result.txt`, con las cuatro peticiones y sus respuestas.

**TC-14 — privacidad por diseño.**

**Procedimiento.** Dos comprobaciones independientes, una a cada lado del
sistema. En la nube se toman **500 muestras** de la columna que identifica
visitantes, se verifica que ninguna tenga forma de dirección MAC, y se recorre el
esquema completo buscando columnas capaces de contener imágenes o datos
personales. En el dispositivo se audita el sistema de archivos completo en busca
de imágenes escritas por el proceso de conteo.

**Criterio de aceptación.** Cero direcciones MAC en claro; ninguna columna con
datos personales; ninguna imagen escrita por el proceso de conteo.

**Resultado.** Las dos comprobaciones son favorables.

*En la nube.* Quinientas muestras, **cero** con formato de dirección
MAC. El recorrido del esquema señala tres columnas, que se examinan una a una:

| Columna señalada | Tipo | Qué contiene realmente |
|---|---|---|
| `telemetry.frame_latency_p50_ms` | real | latencia de cuadro, en milisegundos |
| `telemetry.frame_latency_p95_ms` | real | latencia de cuadro, en milisegundos |
| `wifi_ble_events.visitor_hash` | bytea | el resumen salado y truncado que **sustituye** a la dirección |

Las dos primeras son magnitudes de rendimiento. La tercera es precisamente el
mecanismo de anonimización, no una excepción a él. El aviso del barrido es un
heurístico que se dispara por tipo de dato —cualquier columna binaria— y no un
hallazgo: por eso su veredicto automático queda en «revisar», y la revisión es
la que se acaba de exponer.

*En el dispositivo.* El directorio de salida del extractor de
fotogramas **no existe**, de modo que nunca se escribió uno. La función está
deshabilitada en la configuración de producción y el registro del servicio no la
menciona **ni una vez** desde el 1 de enero de 2026. El barrido del disco
completo encuentra 44 imágenes, todas atribuibles a herramientas de puesta a
punto:

| Ubicación | Cantidad | Origen |
|---|---:|---|
| `/home/pi/calib_archive_20260622/` | 43 | capturas del patrón de calibración, más un diagnóstico de profundidad |
| `/usr/src/people-counter/verify_epipolar.png` | 1 | diagnóstico de alineación de la calibración |

Ninguna proviene del proceso de conteo. El argumento se cierra por construcción:
el servicio corre con el acceso al directorio personal bloqueado y con un
directorio temporal privado, y sus rutas de escritura están limitadas a cuatro
directorios, ninguno de los cuales contiene imagen alguna. **No podría** haber
escrito las de la primera fila aunque lo intentara.

**Procedencia de la evidencia.** `validation/tc14_result.txt`, que reúne el
barrido en la nube y la auditoría del disco tomada sobre el dispositivo en
operación.

**TC-15 — latencia de extremo a extremo.**

**Procedimiento.** Se mide el tiempo entre el instante del hecho y su
persistencia en la base, **por invocación y no por fila**. La distinción es
determinante: el flujo inalámbrico publica un arreglo por ventana de 15 minutos
que inserta muchas filas en una sola ejecución, de modo que contar filas
multiplicaría cada ejecución lenta por el número de dispositivos que llevaba
dentro, y esas filas comparten latencia por construcción —no son observaciones
independientes—. La marca de origen es el instante del evento para conteo y
telemetría, y el cierre de la ventana para el flujo inalámbrico. Se reporta
además la variante que excluye los reenvíos del buffer local, que son latencia
de red del sitio y no de la cadena. Registro completo: **7 059 invocaciones** del
dispositivo piloto.

**Criterio de aceptación.** Percentil 95 no mayor que 5 s.

**Resultado.** La configuración vigente cumple con holgura, sobre 4 728
invocaciones:

| Flujo | Invocaciones | p50 | p95 | Máximo | Por encima de 5 s |
|---|---:|---:|---:|---:|---:|
| Conteo | 880 | 0,213 s | **0,328 s** | 13,9 s | 1,0 % |
| Telemetría | 2 896 | 0,244 s | **0,428 s** | 41,7 s | 4,2 % |
| Inalámbrico | 952 | 0,294 s | **0,435 s** | 31,5 s | 3,1 % |

**El caso detectó además un incumplimiento anterior, y queda documentado.** El
mismo análisis sobre la configuración previa al 8 de junio de 2026 arroja
percentiles muy por encima del umbral:

| Flujo | p50 | p95 | Por encima de 5 s |
|---|---:|---:|---:|
| Conteo | 0,259 s | 6,54 s | 26,2 % |
| Telemetría | 0,280 s | 6,60 s | 34,4 % |
| Inalámbrico | 6,37 s | 7,22 s | 86,7 % |

El veredicto favorable corresponde, por lo tanto, a la configuración desplegada,
y el registro conserva la medición que quedó atrás. Medir por invocación fue lo
que hizo visible el contraste: sobre el recuento por filas, y sobre muestras
pequeñas, la diferencia entre ambas configuraciones se diluye.

**Una invocación programada, posterior a esta medición.** En la cuenta existe
una regla de EventBridge —`people-counter-persist-event-warmup-dev`,
`rate(2 minutes)`— que invoca la función de persistencia con un evento de conteo
duplicado, descartado por la restricción de unicidad, con el solo fin de
mantener tibio su entorno de ejecución. **No forma parte de la plantilla de
infraestructura**: se creó por línea de comandos el **2026-08-06**.

Es **posterior a las mediciones de este caso**, de modo que no explica los
percentiles publicados. Sí implica dos cosas hacia adelante: una corrida nueva
de la medición daría mejor que lo aquí reportado, porque la función ya no parte
en frío; y un despliegue limpio desde la plantilla **no tendría** esa regla, así
que reproduciría el comportamiento sin ella.

**Procedencia de la evidencia.**
`validation/tc15_latency_by_invocation.py` y su salida
`tc15_latency_by_invocation_result.txt`.

**TC-16 — resiliencia ante un corte breve de conectividad.**

**Procedimiento.** Prueba de componente sobre el buffer de salida de producción,
con base temporal: se encolan **treinta eventos** con el enlace caído, se
verifica que queden persistidos, y se simula el restablecimiento drenándolos y
marcando la confirmación de entrega de cada uno.

Condición de corte **fabricada**; el componente ejercitado es el de
producción. El corte físico real se cubre en TC-18.

**Criterio de aceptación.** Los treinta quedan persistidos con el enlace caído y,
al restablecerlo, se drenan íntegros: sin pérdida y sin duplicado.

**Resultado.** Cumple. Treinta encolados y treinta drenados, cero restantes,
cero duplicados, cero perdidos.

**Procedencia de la evidencia.** `validation/tc16_brief.py` y su salida
`tc16_result.txt`.

**TC-17 — resiliencia ante un corte prolongado.**

**Procedimiento.** Mismo banco que TC-16, con el **volumen** correspondiente a 72
horas —no su transcurso—, a las cadencias reales del dispositivo: 864 mensajes de
telemetría a uno cada cinco minutos, 288 del flujo inalámbrico a uno cada quince,
y 150 eventos de conteo. **1 302 mensajes** en total. Un control adicional inyecta
1 500 mensajes contra un tope de 1 000 para comprobar que el mecanismo de
acotamiento actúe.

**Criterio de aceptación.** Los 1 302 persisten y se drenan sin pérdida ni
duplicado, sin desbordar; y el tope acota cuando se lo excede.

**Resultado.** Cumple. Los 1 302 encolados se drenan íntegros —cero restantes,
cero duplicados, cero perdidos— y ninguno se descarta contra el tope configurado
de 50 000. En el control, 1 500 mensajes contra un tope de 1 000 dejan 500
descartados y exactamente 1 000 retenidos.

**Alcance de la evidencia.** Se simula el volumen, no el transcurso. El caso
acredita que la estructura de almacenamiento soporta la cantidad de mensajes de
un corte de 72 horas y los devuelve íntegros; no ejercita la degradación que 72
horas de operación continua podrían producir por otras vías, como el crecimiento
del archivo en disco o la rotación de registros.

**Procedencia de la evidencia.** `validation/tc17_buffer_72h.py` y su
salida `tc17_result.txt`.

**TC-18 — reinicio tras un corte de energía.**

**Procedimiento.** Corte **físico real** de la alimentación —desconexión del riel
de alimentación por Ethernet, no un reinicio por software—, sobre el dispositivo
instalado, el 21 de junio de 2026. Se toma una instantánea del estado antes del
corte y se reconstruye después desde el propio dispositivo, con las marcas de
arranque y del registro del sistema relativas al núcleo, y desde el lado de red.

**Criterio de aceptación.** Reanudación de la operación nominal en menos de 90 s,
con el almacenamiento local íntegro, el sistema de archivos sano y sin pérdida de
publicaciones.

**Resultado.** Primer cuadro procesado a los **46 s** del arranque, con este
desglose:

| Hito | Desde el arranque |
|---|---:|
| Sistema listo (3,1 s de núcleo + 33,8 s de espacio de usuario) | 36,9 s |
| Monitor inalámbrico activo | 10 s |
| Servicio de conteo activo | 36 s |
| Primera conexión con la nube | 37 s |
| **Primer cuadro procesado** | **46 s** |

La verificación de integridad de las dos bases locales devuelve correcto, el
sistema de archivos queda montado en lectura y escritura sin errores en el
registro del núcleo, el servicio no acumula reinicios —arranque limpio, sin
ciclo de caídas— y no hubo limitación por tensión durante la renegociación de la
alimentación. La telemetría se reanudó en la nube después del corte.

**Procedencia de la evidencia.** `validation/tc18_powercut_result.txt`, con
la lista de comandos usados para reconstruir el estado.

**TC-19 — disponibilidad del stack cloud.**

**Procedimiento.** El criterio interroga a **IoT Core,
RDS y Fargate**, de modo que se mide sobre las métricas que cada servicio publica
por su cuenta y no sobre la continuidad de la telemetría del dispositivo: los
huecos de esa serie corresponden a **equipo apagado** —es una unidad de
desarrollo, con traslados off-site entre sesiones— y no informan sobre la
disponibilidad de la nube. Reconstrucción con
`validation/tc19_cloud_availability.py` sobre la ventana de 24 h del 21-06
16:20 al 22-06 16:20 (−03), en intervalos de 5 min.

**Criterio de aceptación.** Disponibilidad no menor que 99 % en la ventana
de 24 h para los tres servicios, sin alarmas disparadas y con los tableros
accesibles.

**Resultado.** Los tres servicios superan el umbral:

| Servicio | Señal | Cobertura |
|---|---|---|
| RDS | `CPUUtilization` / `DatabaseConnections` — la instancia sólo emite mientras corre | **288/288 (100 %)**, CPU 5,0–6,6 %, 9–13,8 conexiones |
| Fargate | `LiveTaskCount` del servicio de Grafana | **288/288 (100 %)**, siempre ≥ 1 tarea viva |
| IoT Core | `Failure` del broker / `PublishIn.Success` | **0 fallos**; publicaciones aceptadas en 287/288 (99,7 %) |

Los tres superan el umbral de 99 %. Las otras dos condiciones del criterio se
reconstruyen por separado, porque el historial de transiciones de alarmas de
CloudWatch está vacío y la accesibilidad de los tableros no quedó registrada:

**Alarmas** (`validation/tc19_alarm_reconstruction.py`). Las métricas que
alimentan a cada alarma sí se retienen, de modo que la condición de las diez se
reevalúa con sus propios parámetros —operador, umbral, períodos de evaluación,
*datapoints-to-alarm* y tratamiento de faltantes—. **Ninguna alcanzó su
condición durante la ventana.** La más exigente, `grafana-tasks` —que trata los
faltantes como incumplimiento— tuvo métrica en los 288 intervalos con un destino
sano en todos.

**Tableros** (`validation/tc19_dashboard_reachability.py`). Grafana corre
detrás de un ALB que publica el estado de sus destinos: `HealthyHostCount` = 1
en **288/288** intervalos, 0 destinos no sanos, **842 peticiones atendidas** y
**cero respuestas 5XX**, con latencia media de 5 ms. La accesibilidad queda
medida, no inferida.

**Procedencia de la evidencia.** Los tres guiones de reconstrucción en
`validation/` —`tc19_cloud_availability.py`,
`tc19_alarm_reconstruction.py` y `tc19_dashboard_reachability.py`— con su salida
unificada en `tc19_result.txt`, sobre métricas publicadas por CloudWatch.

> **Caveat de retención.** CloudWatch conserva la resolución de 5 min durante 63
> días: la ventana de junio deja de ser reconstruible alrededor del **2026-08-23**.
> Por eso la salida del guion se archiva junto al resto de la evidencia.

## 3. Caracterización de banco (requisitos no funcionales)

### Consumo eléctrico

| Estado | Promedio (W) | p95 (W) | Pico (W) | % del presupuesto PoE 25,5 W |
|---|---|---|---|---|
| Pipeline detenido (idle del SO) | 2,16 | 2,19 | 3,06 | 8,5 % |
| Corriendo, escena vacía, idle-throttle ON (~10–16 FPS) | 4,00 | 5,38 | 9,93 | 15,7 % |
| Corriendo, escena vacía, full-throughput (~28 FPS) | 4,67 | 6,57 | 8,75 | 18,3 % |

El consumo es casi independiente de la carga de cómputo (10→28 FPS suma sólo
~0,67 W). Pico absoluto 9,93 W (transitorio), < 40 % del presupuesto.

**Procedencia.** `power_idle.csv`, `power_session.csv` y `power_fullthroughput.csv`, por muestreo del controlador de alimentación del propio equipo; estadísticos con `analyze_bench.py`.


### Térmico (con Active Cooler)

| Métrica | Valor | Límite | Veredicto |
|---|---|---|---|
| Temp CPU máx bajo estrés sostenido | 64,8 °C | 80 °C | ✅ (15 °C de margen) |
| Temp Hailo (telemetría, máx) | 41,8 °C | 85 °C | ✅ |
| Throttling (`get_throttled`) bajo estrés | 0x0 (132 muestras) | 0x0 | ✅ nunca throttleó |

(El comportamiento térmico en gabinete cerrado en las instalaciones se trata
como limitación L2 en §4.)

**Procedencia.** `stress_monitor.csv` (132 muestras bajo estrés sostenido), `analysis_summary.txt` y `soak_system.csv` (soak de 60 min), procesados con `analyze_bench.py`. Las cifras del gabinete cerrado, en `thermal_deploy_uncapped.csv` y `thermal_deploy_capped_1500mhz.csv`.


### Memoria

| Métrica | Valor | Objetivo flota | Veredicto |
|---|---|---|---|
| `memory.peak` del servicio (soak ~60 min) | 416 MiB | < 2048 MiB | ✅ (20 % usado) |
| `memory.peak` bajo stress sintético | 247 MiB | < 2048 MiB | ✅ (12 %) |
| `memory.events` (high/max/oom) y swap | 0 | 0 | ✅ |

El pico bajo estrés sintético es **menor** que el del soak porque el estrés
ejercita procesador y memoria del sistema sin correr el pipeline de visión
completo: el grueso del conjunto de trabajo —los mapas de rectificación y los
búferes de profundidad— sólo se reserva con el pipeline en marcha. La cifra que
dimensiona el equipo es, por lo tanto, la del soak.

**Procedencia.** `analysis_summary.txt` (soak) y `stress_monitor.csv` (estrés), sobre los contadores del grupo de control del servicio; descomposición del conjunto de trabajo en `memory_working_set_result.txt`.


### FPS, throughput y latencia por etapa

| Escenario | FPS efectivo | Etapa dominante |
|---|---|---|
| Escena vacía (SGBM omitido) | 27,8 FPS | inferencia Hailo 15,9 ms + captura 14,1 ms |
| Con persona (SGBM activo) | 21,4 FPS | profundidad SGBM 22,6 ms (~46 % del frame) |

Latencia por frame (telemetría): p50 24,0 ms · p95 26,5 ms. El pipeline es
cámara-bound (~28 FPS) con escena vacía y SGBM-bound (~21 FPS) con gente.
Throughput puro del detector on-chip: **105,9 inf/s** (≈5× el rate del
pipeline → el NPU no es el cuello de botella).

**Procedencia.** `profile_empty_summary.txt` y `profile_empty_perframe.csv`, generados con el modo de perfilado del pipeline y procesados con `parse_profile.py`.


### CPU

| Estado | CPU agregada (4 cores) | FPS |
|---|---|---|
| Nominal, idle-throttle ON | avg 15,3 % · p95 27,4 % | ~10–16 |
| Full-throughput, escena vacía | ~96 % (≈1 core de 4) | ~28 |

El pipeline es serial/single-core; el cuello de botella del FPS es la latencia
serial por frame, no la CPU agregada ni la RAM.

**Procedencia.** `soak_system.csv` (contadores de tiempo de procesador del sistema) y `stress_monitor.csv` (ocupación agregada).


### Sincronización de cámaras (delta L/R)

| Configuración | mediana | p95 | % pares ≤ 5 ms |
|---|---|---|---|
| Sin sync | 0,37 ms | 18,8 ms | 87,8 % |
| Con sync (software, libcamera SyncMode) | 0,025 ms | 0,050 ms | 100,0 % |

El sync por software elimina la cola episódica a ~19 ms (deriva de un período
de frame entre relojes de sensor) sin cableado.

**Procedencia.** `camsync_sin_sync.csv` y `camsync_con_sync.csv` —17 618 pares de marcas temporales de sensor— capturados con `camsync_sin_sync.py` y `camsync_con_sync.py`.


### Cobertura de tests

| Métrica | Valor |
|---|---|
| Tests totales | **1096 pasan y 7 se saltean** (1103 en total) |
| Cobertura global | **81 %** (6 891 sentencias, 1 303 sin cubrir) |
| Módulos críticos | `counter.py` 97 % · `persist_event.py` 98 % · `calibration.py` 92 % · `tracker.py` 91 % · `dedup.py` 82 % |

Los siete saltos son **artefactos de la plataforma de ejecución, no huecos de
cobertura**: cinco ejercitan permisos POSIX que `os.chmod` no expresa en Windows
y dos requieren el filtro WLS de `opencv-contrib`, ausente en la rueda de
Windows. Sobre el sistema operativo del dispositivo los siete se ejecutan.

**Procedencia.** `test_suite_coverage.txt`, corrida sobre el árbol publicado; incluye el motivo declarado de cada salto.


### Foco y calibración estéreo

Las cifras de abajo corresponden a la **calibración que el dispositivo tiene
instalada** —`/etc/people-counter/calibration.npz`, escrita el 2026-06-22 a las
21:08— y al foco de la misma puesta a punto. Es la calibración con la que se
produjo toda la evidencia de campo de §2, incluida la campaña dirigida del 25 de
junio: no hubo recalibración posterior.

| Métrica | Valor | Umbral | Veredicto |
|---|---|---|---|
| Foco — nitidez de centro (izq / der) | 1375 / 1414 | ≥ 200 | ✅ PASS |
| Foco — esquinas (izq / der) | 654 / 694 | ≥ 100 | ✅ PASS |
| Foco — simetría entre lentes | 6,0 % | ≤ 15 % | ✅ PASS |
| Foco — distancia del patrón | 1,50 m | 1,30 – 1,70 m | ✅ PASS |
| Pares con detección válida en ambas cámaras | 36 de 40 (90 %) | ≥ 70 % | ✅ PASS |
| RMS de reproyección estéreo | 0,290 px | ≤ 1 px | ✅ PASS |
| Verificación contra referencia — centro | −2,69 % a 1 765 mm | ≤ 5 % | ✅ PASS |
| Verificación contra referencia — zonas perimetrales | −10,6 % y −26,9 % (caen sobre otros planos: sin distancia declarada) | no entra en el veredicto | — |
| Baseline óptico ‖T‖ | 144,71 mm (+4,71 mm sobre el nominal mecánico de 140 mm) | informativo, ver L4 | — |
| MAE de estatura en geometría cenital real | ≈ 28 mm; dos sujetos (1,68 y 1,82 m), 15/15 dentro de ±10 cm | ±10 cm | ✅ PASS |

**Cómo se llegó a esta calibración.** La sesión del 22 de junio produjo cinco
calibraciones. La verificación de profundidad se hace contra un objetivo
colocado a una distancia medida con cinta, y el veredicto se calcula **solo con
la zona central**, que es la única con distancia declarada. En un espacio
reducido no siempre se puede ubicar el objetivo de modo que llene esa zona a la
distancia elegida, y eso es lo que ocurrió en la primera corrida: la zona
central quedó abarcando más de un plano —desviación de 348 mm, la más alta de
las tres— mientras que **las dos zonas perimetrales medían −2,76 % y +0,28 %
contra esa misma referencia de 1 620 mm**. Es decir, la calibración estaba
midiendo bien; lo que no estaba bien era dónde caía la zona que decide el
veredicto. Es exactamente la situación que las etiquetas por zona del reporte
existen para distinguir, y la razón por la que el verdict se acompaña de ellas.

Se repitió la toma hasta obtener una captura en la que el objetivo llenara la
zona central, y esa es la calibración instalada: su lectura de centro es la más
limpia de la sesión —desviación de 5,4 mm con 100 % de relleno— y verifica en
−2,69 %. En ella son las zonas perimetrales las que caen sobre otros planos, con
el patrón inverso.

Las tomas de esa sesión se hicieron con el **modo de baja luz** del asistente,
pensado justamente para espacios chicos y luz difícil: relaja las guardas de
calidad —exposición, desenfoque, nitidez de esquina y balance entre cámaras— y
emite una advertencia genérica de no confiar en la calibración resultante para
profundidad. Esa advertencia se imprime siempre que se usa el modo, con
independencia del resultado: es una precaución de la herramienta, no una
medición sobre esta calibración. El registro completo de la corrida está
publicado para que pueda comprobarse.

**Por eso el respaldo de la profundidad no es el informe de calibración sino
TC-08**, que la mide donde importa y contra una referencia independiente:
quince estaturas tomadas con cinta, dos sujetos, todas dentro de ±10 cm y con
error absoluto medio de 2,8 cm, sobre esta misma calibración y en la geometría
de montaje real. El RMS y la verificación de zona son indicadores de que el
solve no está degenerado; la exactitud útil la establece TC-08.

**Procedencia.** La calibración misma se publica como **adjunto de la versión
etiquetada**: `calibration.npz`, SHA-256 `4cb38d080906…`,
byte-idéntica a la que el dispositivo tiene instalada. Con ella pueden
recomputarse de forma independiente el baseline, la alineación entre lentes y
los mapas de rectificación, sin depender de ningún informe. El registro de la
corrida que la produjo está en `validation/calib_run.log` y su
verificación contra distancia medida en `validation/calib_gt.json`;
el foco, en `validation/focus_report.txt`.

Los informes HTML del asistente **no se versionan**: son ayudas para el
operador durante la puesta a punto —llevan imágenes del espacio físico— y no
aportan nada que no esté en el archivo de calibración o en esos registros.

## 4. Hallazgos de la validación y mitigaciones

La validación (banco + TC dirigidos + validación dirigida en las instalaciones)
arrojó **cinco hallazgos accionables**: cuatro resueltos/mitigados durante el
desarrollo —desfase de sincronización estéreo, compresión de estatura (L1),
consumo de CPU en escena vacía y margen térmico en gabinete cerrado (L2)— y uno
caracterizado como limitación (sub-conteo en cruces simultáneos opuestos, TC-03).
La tabla los resume; el detalle de las limitaciones sigue debajo.

| Hallazgo | Detectado en | Causa | Mitigación | Estado |
|---|---|---|---|---|
| Compresión de estatura a mount bajo (L1) | TC-08 (estatura) | edge-bleed near-camera en el extractor de head-depth a mount ~2,4 m | techo `max_head_height` + ajuste SGBM (uniqueness / WLS) | **Mitigado** — validado A/B, dentro de ±10 cm |
| Desfase estéreo episódico ~19 ms (~12 % de pares) | caracterización de sync L/R | captura en libre corrida: las fases de ambos sensores derivan | captura sincronizada por software (converge + hold) | **Resuelto** — p95 0,05 ms, 100 % < 5 ms |
| Sub-conteo en cruces simultáneos en sentidos opuestos | TC-03 dirigido (2 corridas) | no aislada con el registro disponible; el modo observado es la pérdida de **uno** de los dos cruces de cada par, sin fusión de identidades | canary `ambiguous_reject_count`, incorporado después para observar la hipótesis en producción | **Limitación documentada** |
| Throttling térmico en gabinete cerrado (L2) | validación en las instalaciones | heat-soak del gabinete cerrado — el gabinete, no el disipador, es el factor dominante; el 24-06, sin límite, llega a 84,25 °C, el firmware **activa** la protección térmica y el reloj cae de 2400 a 2311 MHz | freq-cap 1500 MHz (sin pérdida de FPS por ser Hailo-bound); ranuras de ventilación ampliadas como medida adicional opcional | **Mitigado** — el 25-06, con cap y bajo la carga de la campaña dirigida: media 73,9 °C, máximo 82,6 °C sobre 233 muestras y **cero** con el límite térmico activo (banco: máx 64,8 °C) |
| CPU alta en escena vacía (~162 % agregado) | caracterización de CPU / profiling de banco | conversión de color redundante por frame + rectificación derecha eager + parsing de tramas sin pre-filtro + sin FPS adaptativo | eliminación de la conversión + rectificación lazy + pre-filtro de tramas + `vision.idle_throttle` (10 FPS en escena vacía, wake instantáneo) | **Resuelto** — ~50 % en escena vacía, count-neutral (validado de extremo a extremo en las instalaciones) |

### Limitaciones conocidas (detalle)

**L1 — Compresión de estatura a mount bajo (2,413 m) — MITIGADA.** A esta
altura las cabezas quedan a 0,6–0,73 m, donde la diferencia de estatura entre
personas se traduce en muy poca disparidad y el pipeline de runtime
(SGBM `downscale=4` + WLS + extracción por slices de 10 cm) la suaviza,
capturando sólo ~10 % de la variación real. No afecta el conteo (usa centroide
2D), sólo la segmentación por rango de estatura. Mitigada por config —
`max_head_height_m: 1.95`, `uniqueness_ratio: 15`, WLS `λ: 1500`— y validada
A/B sobre la misma persona en las instalaciones (sujeto de 1,82 m: de ~1,73 m / −9 cm a
~1,78 m / −4 cm, dentro de ±10 cm, sin costo de FPS). Residual ~−4 cm; su fix
de raíz (estimador robusto del crown) queda fuera del alcance del prototipo.

**L2 — Margen térmico en gabinete cerrado — MITIGADA.** La limitación es sobre
el comportamiento del **gabinete impreso cerrado en uso real**. Las dos jornadas
en que el dispositivo estuvo montado en su emplazamiento, dentro del gabinete,
son el **24 de junio** —día del montaje— y el **25**, en el que se ejecutó la
campaña dirigida. Las cifras de abajo se acotan a esas dos jornadas, y no al
promedio de los archivos completos, que abarcan momentos en otras condiciones.

**24 de junio — sin límite de frecuencia.** Ya montado en el emplazamiento, la
temperatura sube de 60,1 °C a **84,25 °C** entre las 14:30 y las 14:47. En el
pico el firmware **activa** el
límite térmico y **la frecuencia cae de 2400 a 2311 MHz**: el throttling no se
infiere de una bandera, se ve ocurrir en el registro. Es la única muestra de toda
la serie con el bit activo. Ese mismo día se aplica el límite de frecuencia, que
ya figura en 1500 MHz hacia el final de la jornada.

**25 de junio — con límite a 1500 MHz, y bajo la carga de los ensayos.** Es la
jornada más exigente del período con límite, porque es cuando se ejecutó la
campaña: 233 muestras de telemetría, `arm_clock_mhz` en 1500 en todas.

| | Valor |
|---|---:|
| Temperatura media | **73,9 °C** |
| Máximo | **82,6 °C** |
| Muestras con el límite térmico **activo** | **0** |

Las banderas que aparecen ese día son **históricas**, arrastradas del episodio
del 24: indican que hubo throttling en algún momento previo, no que lo haya
durante los ensayos. Con 82,6 °C de pico y el límite de frecuencia aplicado, el
firmware no volvió a activar la protección térmica.

Que sean del día anterior no es una conjetura: **el 24 y el 25 son la misma
sesión**. El dispositivo arrancó el 24 a las 14:24 y siguió encendido hasta las
19:06 del 25, veintiocho horas y media sin reiniciar, con el equipo montado en
su emplazamiento durante todo ese lapso. En esa única sesión entra el pico de
84,25 °C con la protección activándose, la aplicación del límite de frecuencia y
la campaña dirigida completa. Las banderas no se limpian hasta el siguiente
arranque, que fue el 26 a las 09:20 — y desde ahí figuran en cero.

> **Qué cubren los archivos publicados y qué no.**
> `thermal_deploy_uncapped.csv` reúne 1 162 muestras en nueve jornadas entre el 7
> y el 24 de junio, de las cuales sólo la última corresponde al gabinete en el
> emplazamiento; los días previos, con el equipo fuera, no superan los 63,9 °C.
> `thermal_deploy_capped_1500mhz.csv` contiene 1 272 muestras en **dos bloques
> separados por 35,9 días sin registro**, y **ninguno de los dos corresponde al
> emplazamiento**: 687 muestras del 26 al 28 de junio y 585 del 3 al 5 de agosto,
> ambos con el equipo ya retirado y operando en el puesto de desarrollo. Su media
> de 64,2 °C y su máximo de 67,75 describen el régimen del límite de frecuencia
> **fuera** de la condición que esta limitación caracteriza, con una carga muy
> baja —2 y 5 eventos de conteo en tres días— y a unos diez grados por debajo de
> lo que el mismo límite dio en el emplazamiento bajo la carga de los ensayos.
>
> **Ninguno de los dos cubre las jornadas del emplazamiento**: el de sin límite
> termina el 24 y el de con límite empieza el 26, de modo que el 25 —el día de la
> campaña— no está en ninguno. Por eso se publica una tercera serie,
> `validation/thermal_deploy_onsite.csv`, con la **sesión completa del
> emplazamiento**: 349 muestras entre las 14:30 del 24 y las 19:06 del 25, en el
> mismo formato de columnas que las otras dos. Promediar el archivo con límite
> entero da 63,3 °C, que no describe ninguna situación real.

**Mitigación adoptada:** limitar la frecuencia máxima del procesador a 1500 MHz
(`cpu-freq-cap.service`, persistente; sin pérdida de FPS por ser el pipeline
Hailo-bound). El contraste entre los dos días es directo: la misma condición de
gabinete cerrado pasa de **cruzar el umbral de protección térmica** a operar bajo
la carga de los ensayos sin activarla. En banco, con disipador activo y sin
gabinete, el máximo fue 64,8 °C.

Por eso el límite de frecuencia se adopta como **mitigación definitiva** y no
como un parche a revertir; la **ampliación de las ranuras de ventilación queda
como medida adicional, no requerida** en esta configuración.

*Procedencia: `validation/thermal_deploy_onsite.csv` para las dos jornadas del
emplazamiento —349 muestras de la sesión, de las cuales 233 corresponden al 25 y
reproducen las cifras citadas—; `validation/thermal_deploy_uncapped.csv` y
`validation/thermal_deploy_capped_1500mhz.csv` para las series fuera del
emplazamiento.*

**L3 — Alcance de la validación de conteo.** La evidencia de conteo es de
**carácter indicativo**: una unidad, un operador, muestras del orden de la decena
por caso y —el dato más acotante— **una única sesión continua**. Los ocho casos
de campo, de TC-01 a TC-08, se ejecutaron entre las 14:24 del 24 de junio y las
19:06 del 25, con el dispositivo montado en su emplazamiento y sin un solo
reinicio de por medio: 28,7 horas. No hay repetición en otra jornada, con otra
puesta a punto ni en otras condiciones de luz, de modo que la evidencia no
distingue entre el comportamiento del sistema y las particularidades de esa
sesión.

Los demás casos no dependen del montaje y por eso no comparten esa restricción:
TC-09, TC-10, TC-16 y TC-17 son pruebas de componente sobre bases temporales;
TC-11, TC-12, TC-13 y TC-19 se ejecutan contra la nube; TC-14 combina un barrido
en la nube con una auditoría del disco del equipo; TC-15 abarca 19 jornadas de
actividad; y TC-18 es un corte físico de alimentación del 21 de junio, anterior
al montaje. La caracterización de banco de §3 es también del 21 de junio, fuera
del emplazamiento. No es una limitación del producto sino de la fuerza estadística
de la evidencia. Los casos dirigidos controlados la elevan —TC-01, TC-02, TC-04,
TC-05 y TC-07 con sus eventos persistidos tabulados uno a uno, TC-08 con dos
sujetos y quince mediciones, TC-06 con alturas medidas por el propio sistema—,
pero ninguno alcanza el tamaño de muestra necesario para una medición de
exactitud estadísticamente robusta, que corresponde a la etapa de piloto
(futura). El único caso que reveló una limitación funcional real es el
simultáneo estricto bidireccional (ver TC-03); TC-06 no alcanza su criterio por
la limitación descrita en L5.

**L4 — Baseline óptico ≠ nominal mecánico.** La calibración estima el baseline
entre los centros ópticos (pupilas de entrada del fisheye), que difiere del
nominal mecánico de 140 mm por la óptica y las tolerancias de impresión del
case: 144,6 ± 2,7 mm sobre n=7 calibraciones. Sin impacto (el depth quedó
validado por ground-truth y el conteo no usa el baseline directamente); el gate
del baseline se degradó a informativo para prevenir un false-FAIL — el verdict
de calibración lo decide el ground-truth de profundidad.

**L5 — La eficacia del filtro geométrico se degrada cerca de su umbral.** El
filtro anti-falsos-positivos por altura compara la **estatura mediana estimada**
de la trayectoria contra un umbral fijo, 1,00 m en la configuración evaluada. Su
eficacia depende de que la dispersión del estimador sea pequeña frente al margen
entre el objeto y el umbral, y esa condición deja de cumplirse cuando el objeto
se aproxima al umbral: la estatura estimada arrastra la dispersión descrita en
**L1** —compresión y ruido del extractor de profundidad a montaje bajo—, de modo
que cerca del corte el error de medición y el margen son del mismo orden y el
resultado del filtro se vuelve incierto.

**Alcance, con las cifras del propio registro.** Las tres pasadas que el filtro
rechazó midieron 0,96, 0,96 y 0,97 m: un margen de **3 a 4 cm** respecto del
umbral de 1,00 m. Ese margen hay que leerlo contra la exactitud del estimador
que establece TC-08 sobre el mismo montaje: **error absoluto medio de 2,8 cm,
con un máximo observado de 6 cm**. Margen y error son del mismo orden, y el
máximo observado supera al margen — de ahí que el resultado del filtro se
vuelva incierto en esa franja.

El sustituto que TC-06 emplea por necesidad —una persona atravesando gateando—
queda **inmediatamente** por debajo del umbral y no holgadamente por debajo, de
modo que el ensayo ejercita el filtro en la parte menos favorable de su rango.
Eso explica tanto los tres rechazos correctos como la cuarta pasada, que superó
el umbral y no correspondía rechazar.

**Sin mitigación adoptada.** Subir el umbral reduciría el margen frente a
personas de baja estatura, que es el error caro; bajarlo dejaría pasar objetos
que hoy se rechazan. La vía de mejora no es mover el corte sino reducir la
dispersión del estimador de estatura, que es lo que L1 acota.
