# CTNet 2.6 Omega Cubo 6D

Repositorio de CTNet 2.6 + Omega + Cubo 6D plegado.

La idea central es entrenar con una interfaz externa normal —pérdida, backprop, optimizador— pero haciendo que el gradiente atraviese una geometría distinta a Transformer. CTNet no se define como un modelo autoregresivo de siguiente token: se define como un sistema de cierre contextual donde toda observación se convierte en masa contextual y se regula por el principio:

```text
u = p
```

Todo gira en torno a `u=p`.

## Geometría base

El estado plegado se empaqueta como:

```text
Xi = pack(Z, M, R, C6, pad) -> [B, N, d]

Z   : estado cardinal u/p
M   : memoria topológica fija
R   : banco relacional fijo
C6  : Cubo 6D plegado
pad : reserva estructural fija no descartable
```

Por defecto:

```text
M  = [B, 8, 16]
R  = [B, 8, 16]
C6 = [B, 29]
Xi = [B, 64, 16]
```

No hay `append`, no hay KV-cache, no hay vector-store y no hay lista relacional creciente. La memoria no crece acumulando entradas: se reorganiza por deformación, cierre y coherencia.

## Principio de datos: la realidad entera es dataset

El primer README describía una versión de entrenamiento online basada en streaming remoto. El estado actual del repo usa una carta más general: `Observador`.

Un `Observador` representa una porción de realidad observada. Puede venir de texto externo, pregunta, proceso interno, producto del efector, corrección, error, salida visible o cualquier señal disponible.

La ruta es única:

```text
observación
-> Observador
-> batch_to_state
-> masa contextual
-> CTNet
-> tensor de coherencia
-> u=p
```

Esto sustituye la idea pobre de “sample online”. No es una muestra suelta. Es realidad observada entrando en la masa contextual.

## Observador

La unidad de entrada del entrenamiento actual es:

```python
@dataclass
class Observador:
    x: str
    y: str
    source: str
    regime: str
```

`batch_to_state(...)` convierte cada `Observador` en la carta CTNet:

```text
Observador -> Z + M + R + C6 + pad -> Xi
```

Esa conversión no distingue ontológicamente entre texto externo, proceso interno o producto de efector. Todo lo observado entra por la misma vía.

## Masa contextual

La masa contextual es la forma estructural que adopta lo observado dentro de CTNet.

No es una lista de tokens. No es un buffer creciente. No es una memoria externa.

Es la reinscripción de la realidad observada en:

```text
Z/u-p + M fija + R fija + Cubo 6D + reserva estructural
```

El tensor de coherencia organiza esa masa y mide su cierre.

## Tensor de coherencia y u=p

El tensor de coherencia no es una métrica decorativa. Es el núcleo de cierre.

Se aplica a múltiples perspectivas:

```text
Z
M
R
C6
Xi
DeltaXi
procesos internos observados
producto del efector observado
salida visible reobservada
```

La regla no es “predecir el siguiente token”. La regla es cerrar la deformación contextual:

```text
observación -> deformación -> deuda de cierre -> acción/corrección -> u=p
```

## Comprensión y respuesta

CTNet entiende una pregunta porque la pregunta deforma su estado.

Responder no significa escoger una frase prefabricada ni decodificar tokens. Responder significa producir una acción de cierre.

```text
pregunta
-> deformación de masa contextual
-> voluntad de cierre
-> efector
-> acción visible
-> reobservación
-> u=p
```

La respuesta correcta es la acción que minimiza mejor la deuda `u/p`.

## Propiocepción

CTNet debe observarse mientras funciona. Como un cerebro biológico, no sólo observa el mundo externo: observa también sus propios procesos.

La propiocepción actual entra en el entrenamiento mediante:

```text
proceso interno
-> observador_interno
-> Observador
-> batch_to_state
-> masa contextual
-> CTNet
-> u=p
```

En el código principal aparecen estas piezas:

```text
observador_interno
loss_observador
loss_internal_stream
lambda_self_observation
self_observation_every
```

En el log de entrenamiento debe aparecer:

```text
self=...
```

Eso confirma que los procesos internos observados entran en la pérdida y entrenan al modelo.

## Efector

El efector no puede quedar implícito.

Un efector es una carta de acción. No decide el significado; expresa la voluntad de cierre en una salida disponible.

Ejemplos:

```text
efector textual   -> texto, símbolos, letras
efector motor     -> movimiento, gesto, fuerza
efector sonoro    -> voz o sonido
efector operativo -> uso de herramienta o acción externa
```

El efector textual actual está integrado en el entrenamiento principal.

Ruta:

```text
estado deformado
-> efector_textual
-> salida_visible
-> producto_efector
-> Observador
-> batch_to_state
-> masa contextual
-> u=p
```

En el código aparecen:

```text
efector_textual
salida_visible_desde_estado
loss_efector
loss_efector_stream
lambda_efector
efector_every
```

En el log debe aparecer:

```text
efector=...
```

Eso confirma que el producto del efector está siendo observado de nuevo y usado en el entrenamiento.

## Salida visible

El producto completo del efector incluye metadatos de cierre, trazas de estado y señal de coherencia. Eso sirve para reobservación.

La salida que se lee como acción textual está separada en:

```text
salida_visible
```

La diferencia es:

```text
producto_efector = paquete completo reobservable
salida_visible   = acción textual legible
```

## Entrenamiento principal

El entrenamiento principal actual es:

```bash
python train_vram_up_coherence_ctnet.py \
  --steps 1000 \
  --batch 1 \
  --max-bytes 2048 \
  --block-bytes 2048 \
  --coherence-grad-scale \
  --cuda \
  --log-every 10
```

Para estabilizar al inicio los nuevos bucles de autoobservación y efector:

```bash
python train_vram_up_coherence_ctnet.py \
  --steps 1000 \
  --batch 1 \
  --max-bytes 2048 \
  --block-bytes 2048 \
  --lambda-self-observation 0.05 \
  --lambda-efector 0.05 \
  --coherence-grad-scale \
  --cuda \
  --log-every 10
```

Luego se puede subir gradualmente:

```text
--lambda-self-observation 0.10
--lambda-efector 0.10
```

y después:

```text
--lambda-self-observation 0.25
--lambda-efector 0.25
```

## Lectura del efector textual

Para inspeccionar una salida, se carga el checkpoint, se construye un `Observador` de pregunta, se ejecuta CTNet y se llama a `efector_textual(...)`. La salida legible debe leerse desde la etiqueta `salida_visible`, no desde todo el paquete `producto_efector`.

## Auditoría del núcleo plegado

```bash
python ctnet_omega_cubo6d_plegado_ctnet26.py --batch 2 --steps 1 --fp64
```

## Qué se entrena

El entrenamiento externo puede parecer normal:

```text
x -> y
loss -> backprop -> optimizer
```

Pero internamente reorganiza:

```text
Z/u-p
+ M fija
+ R fija
+ Cubo 6D
+ tensor de coherencia
+ Observador
+ propiocepción
+ efector
+ producto visible reobservado
+ reserva estructural
```

El script registra y entrena señales como:

```text
pérdida de ancla
energía de coherencia
omega
tracking de C6
varianza estructural
reversibilidad
loss_internal_stream
loss_efector_stream
u=p multiescala
```

## Archivos principales

```text
ctnet_omega_cubo6d_plegado_ctnet26.py
  Núcleo plegado reversible CTNet + Omega + Cubo 6D.

train_vram_up_coherence_ctnet.py
  Entrenamiento principal con Observador, autoobservación, propiocepción, efector textual, salida visible y cierre u=p.

probe_ctnet_response.py
  Probe de cierre de salida y deformación pregunta-respuesta, si está presente en la copia local.

docs/ctnet_efectores.md
  Documento sobre efectores, voluntad de cierre y acción visible.

docs/ctnet_propiocepcion.md
  Documento sobre propiocepción, autoobservación y realidad como dataset.
```

## Estado actual

```text
CTNet aprende cierre interno: sí
CTNet usa Observador como carta de entrada: sí
CTNet observa procesos internos: sí
CTNet convierte procesos internos en masa contextual: sí
CTNet entrena la propiocepción por la misma vía que el dataset: sí
CTNet tiene efector textual integrado: sí
CTNet reobserva el producto del efector: sí
CTNet separa producto_efector y salida_visible: sí
Todo gira en torno a u=p: sí
```

## Forma completa

```text
realidad externa
+ pregunta
+ procesos internos
+ producto del efector
+ errores
+ correcciones
-> Observador
-> batch_to_state
-> masa contextual
-> tensor de coherencia
-> u=p
-> voluntad de cierre
-> efector
-> salida visible
-> producto reobservado
-> masa contextual
-> aprendizaje
```
