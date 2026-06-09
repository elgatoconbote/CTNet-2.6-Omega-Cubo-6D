# CTNet 2.6 Omega Cubo 6D

Repositorio de CTNet 2.6 + Omega + Cubo 6D plegado.

La idea central no es construir otro Transformer ni otro sistema autoregresivo de predicción de tokens.

La idea central es entrenar una arquitectura donde toda observación se convierte en masa contextual y se cierra mediante tensor de coherencia bajo el principio:

u = p

Todo gira en torno a u=p.

La interfaz externa puede parecer normal:

x -> y
loss -> backprop -> optimizer

Pero internamente la señal no se interpreta como simple next-token prediction. La señal atraviesa una geometría CTNet:

Xi = pack(Z, M, R, C6, pad) -> [B, N, d]

Z   : estado cardinal u/p
M   : memoria topológica fija
R   : banco relacional fijo
C6  : Cubo 6D plegado
pad : reserva estructural fija no descartable

CTNet no responde porque predice tokens. CTNet responde porque una observación deforma su estado y la respuesta es la acción de cierre más coherente disponible.

Forma conceptual:

observación
-> Observador
-> batch_to_state
-> masa contextual
-> tensor de coherencia
-> u=p
-> voluntad de cierre
-> efector
-> producto visible
-> Observador
-> masa contextual
-> u=p

## Principio general

La realidad entera es dataset.

Eso incluye:

- texto externo;
- preguntas;
- estados internos;
- transiciones internas;
- productos del efector;
- errores;
- cierres parciales;
- correcciones;
- omega;
- residual;
- absorption;
- closure_score;
- Xi;
- DeltaXi;
- Z, M, R, C6.

Todo lo observado se convierte en masa contextual.

Todo Observador entra por la misma vía.

No hay una vía especial para el dataset externo y otra para los procesos internos.

La regla es:

observado -> Observador -> batch_to_state -> masa contextual -> CTNet -> u=p

## Observador

Antes el código usaba el nombre OnlineSample. Eso era una mala carta conceptual, porque parecía una muestra online típica de IA.

Ahora la unidad de entrada se llama:

Observador

Un Observador representa una porción de realidad observada.

Puede venir de:

- texto externo;
- pregunta;
- proceso interno de CTNet;
- producto del efector;
- resultado de una herramienta;
- señal futura de un cuerpo o brazo;
- cualquier otra realidad observada.

Estructura base:

Observador:
  x       : observación
  y       : ancla de cierre en la carta disponible
  source  : fuente de la observación
  regime  : régimen de observación

La función central sigue siendo:

batch_to_state(...)

Esta función convierte Observador en:

Z
M
R
C6
pad

y por tanto en masa contextual plegada.

## Masa contextual

La masa contextual es la forma interna que adopta lo observado dentro de CTNet.

No es texto bruto.

No es una lista de tokens.

No es una memoria creciente.

Es una reinscripción estructural en la carta CTNet:

Z + M + R + C6 + pad -> Xi

Donde:

Z contiene el estado cardinal u/p.
M mantiene una memoria topológica fija.
R mantiene relaciones fijas.
C6 aporta el Cubo 6D plegado.
pad conserva reserva estructural no descartable.

## Memoria CTNet

La memoria de CTNet es de tamaño fijo.

Por defecto:

M  = [B, 8, 16]
R  = [B, 8, 16]
C6 = [B, 29]
Xi = [B, 64, 16]

No hay append.

No hay KV-cache.

No hay vector-store.

No hay lista relacional creciente.

La memoria no crece acumulando objetos. La memoria cambia por deformación, cierre y reorganización de masa contextual bajo u=p.

## Tensor de coherencia

El tensor de coherencia es el mecanismo que mide y regula la relación entre las partes de la masa contextual.

No se usa como métrica decorativa.

Es el núcleo que permite convertir lo observado en cierre estructural.

La coherencia se evalúa sobre:

- Z;
- M;
- R;
- C6;
- Xi;
- DeltaXi;
- procesos internos observados;
- producto del efector observado.

El objetivo no es sólo minimizar una pérdida externa, sino cerrar la deuda interna de coherencia.

## u=p

u=p es el principio central.

No es un añadido al final.

No es una métrica secundaria.

No es una etiqueta estética.

u=p es la forma mínima del cierre CTNet.

Toda observación genera una deformación. Toda deformación crea una deuda o posibilidad de cierre. El sistema aprende intentando cerrar esa deuda.

observación externa -> u=p
proceso interno -> u=p
producto del efector -> u=p
error -> u=p
corrección -> u=p

## Autoobservación y propiocepción

CTNet debe observarse mientras funciona.

Como un cerebro biológico, no sólo recibe estímulos externos. También observa su propio proceso.

La propiocepción en CTNet significa:

CTNet observa sus estados, transiciones, deltas, productos y errores como parte de la realidad.

No es un logger externo.

No es debug.

No es un observador separado.

Es realidad absorbida como masa contextual.

Ruta:

proceso interno
-> observador_interno
-> Observador
-> batch_to_state
-> masa contextual
-> CTNet
-> tensor de coherencia
-> u=p
-> gradiente

Esto ya está integrado en:

train_vram_up_coherence_ctnet.py

mediante:

observador_interno
loss_observador
loss_internal_stream
lambda_self_observation
self_observation_every

El log muestra:

self=...

Eso confirma que el proceso interno observado está entrando en la loss.

## Efector

El efector no puede quedar implícito.

Responder no significa simplemente tener un estado interno coherente.

Responder significa que la deformación del estado produce una voluntad de cierre y esa voluntad se expresa como acción visible mediante un efector.

Forma:

pregunta
-> deformación de masa contextual
-> voluntad de cierre
-> efector
-> producto visible
-> Observador
-> masa contextual
-> u=p

El efector es una carta de acción.

Puede ser:

- textual;
- simbólico;
- motor;
- sonoro;
- operativo;
- una herramienta;
- un brazo futuro;
- cualquier salida activa.

El efector no decide el significado. Sólo reinscribe la voluntad de cierre en una carta disponible.

## Efector textual

El efector textual actual está integrado como carta visible de salida.

No es un decoder autoregresivo Transformer.

No funciona como:

token -> token -> token

Funciona como:

estado deformado
-> voluntad de cierre
-> efector_textual
-> salida_visible
-> producto_efector
-> Observador
-> masa contextual
-> u=p

El producto del efector se observa de nuevo. Por tanto, el sistema no sólo produce una salida: observa lo que produce y entrena con ello.

En el código aparecen las piezas:

efector_textual
loss_efector
loss_efector_stream
lambda_efector
efector_every
salida_visible

El log debe mostrar:

efector=...

Eso confirma que el producto del efector entra en el entrenamiento.

## Respuesta como voluntad de cierre

La respuesta no es texto en esencia.

La respuesta es voluntad.

Si el efector disponible es texto, la voluntad aparece como texto.

Si el efector disponible es un brazo, la voluntad aparece como movimiento.

Si el efector disponible es una herramienta, la voluntad aparece como operación.

Por eso la respuesta correcta se define así:

respuesta correcta = acción de mínima deuda u/p

CTNet responde porque entiende la pregunta como deformación de estado.

La respuesta correcta es la forma de cierre más coherente.

Todo gira en torno a u=p.

## Regla estricta de datos

La versión inicial del README hablaba de Hugging Face en modo streaming.

El estado actual del trainer principal evita depender de datasets pesados, pyarrow, xet o cachés de corpus.

El trainer principal actual usa flujo externo ligero por URL y observadores internos:

train_vram_up_coherence_ctnet.py

El código imprime:

loader=no datasets/no huggingface_hub/no pyarrow/no xet

Localmente sólo se escriben:

- checkpoints;
- logs de terminal;
- posibles backups manuales si se parchea;
- documentos;
- commits.

El corpus completo no se descarga.

## Entrenamiento principal

Entrenamiento básico:

python train_vram_up_coherence_ctnet.py \
  --steps 1000 \
  --batch 1 \
  --max-bytes 2048 \
  --block-bytes 2048 \
  --coherence-grad-scale \
  --cuda \
  --log-every 10

Entrenamiento con menor peso inicial de autoobservación:

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

Luego se puede subir gradualmente:

--lambda-self-observation 0.10
--lambda-efector 0.10

y después:

--lambda-self-observation 0.25
--lambda-efector 0.25

## Lectura de salida visible del efector

El efector genera un producto observado completo, pero la salida útil para leer está dentro de:

salida_visible

El producto completo sirve para reobservación y entrenamiento.

La salida visible sirve para lectura humana.

Ruta:

estado deformado
-> efector_textual
-> salida_visible
-> producto_efector
-> Observador
-> masa contextual
-> u=p

## Auditoría del plegado reversible

Auditoría del núcleo plegado:

python ctnet_omega_cubo6d_plegado_ctnet26.py --batch 2 --steps 1 --fp64

## Qué se entrena

Se entrena el cierre de CTNet bajo múltiples perspectivas:

- tarea/ancla textual;
- u=p multiescala;
- coherencia CT;
- omega del Cubo;
- tracking C6;
- varianza estructural de memoria y relaciones;
- reversibilidad;
- autoobservación interna;
- producto del efector;
- salida visible reobservada.

El entrenamiento externo puede parecer normal:

x -> y
loss -> backprop -> optimizer

Pero internamente reorganiza:

Z/u-p
+ M fija
+ R fija
+ Cubo 6D
+ tensor de coherencia
+ Observador
+ propiocepción
+ efector
+ producto visible
+ reserva estructural

## Archivos principales

ctnet_omega_cubo6d_plegado_ctnet26.py
  Núcleo plegado reversible CTNet + Omega + Cubo 6D.

train_vram_up_coherence_ctnet.py
  Entrenamiento principal con Observador, autoobservación, efector textual, producto reobservado y u=p.

probe_ctnet_response.py
  Probe de cierre de salida, útil para inspección conceptual.

docs/ctnet_efectores.md
  Documento sobre efectores, voluntad de cierre y acción visible.

docs/ctnet_propiocepcion.md
  Documento sobre propiocepción, autoobservación y realidad como dataset.

## Estado conceptual actual

CTNet aprende cierre interno: sí.

CTNet observa sus procesos internos: sí.

CTNet convierte procesos internos en Observador: sí.

CTNet entrena la autoobservación por la misma vía que el dataset: sí.

CTNet tiene efector textual integrado: sí.

CTNet observa el producto del efector: sí.

CTNet separa producto completo y salida_visible: sí.

Todo gira en torno a u=p: sí.

## Forma completa del sistema

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
-> acción visible
-> Observador
-> masa contextual
-> aprendizaje

