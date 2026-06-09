# CTNet 2.6 Omega Cubo 6D

Repositorio de entrenamiento online para **CTNet 2.6 + Omega + Cubo 6D plegado**.

La idea central es entrenar con una interfaz externa normal —pérdida, backprop, optimizador— pero hacer que el gradiente atraviese una geometría distinta a Transformer:

```text
Xi = pack(Z, M, R, C6, pad) -> [B, N, d]

Z   : estado cardinal u/p
M   : memoria topológica fija
R   : banco relacional fijo
C6  : Cubo 6D plegado
pad : reserva estructural fija no descartable
```

## Regla estricta de datos

El corpus **no se descarga completo**. El entrenamiento usa Hugging Face en modo streaming:

```python
load_dataset(..., streaming=True)
```

Eso consume ejemplos online conforme se iteran. Localmente sólo se escriben:

- checkpoints;
- métricas JSONL;
- configuración de corrida;
- logs.

## Datasets online incluidos

La mezcla por defecto es:

```text
55% HuggingFaceFW/fineweb-edu, sample-10BT
25% open-web-math/open-web-math
15% AI-MO/NuminaMath-CoT, filtrado para excluir synthetic_* y orca_math
 5% princeton-nlp/SWE-bench
```

Cada ejemplo se normaliza online a:

```json
{
  "x": "condición fuente",
  "y": "objetivo / solución / patch",
  "regime": "tipo de régimen",
  "loss_mode": "modo de entrenamiento externo",
  "source": "dataset remoto",
  "meta": {}
}
```

## Memoria CTNet

Sí: la memoria de CTNet es de **tamaño fijo**.

Por defecto:

```text
M = [B, 8, 16]
R = [B, 8, 16]
C6 = [B, 29]
Xi = [B, 64, 16]
```

No hay `append`, no hay KV-cache, no hay vector-store, no hay lista relacional creciente. La forma de `M` y `R` se audita durante el entrenamiento.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si vas a usar datasets gated o con condiciones de acceso:

```bash
huggingface-cli login
```

## Preview del stream online

Esto no descarga el corpus. Sólo toma unas muestras remotas:

```bash
python train_streaming_ctnet.py --preview --preview-n 3
```

## Entrenamiento básico online

```bash
python train_streaming_ctnet.py \
  --steps 1000 \
  --batch 2 \
  --out-dir runs/online_stream \
  --coherence-grad-scale
```

## Entrenamiento sólo matemático online

```bash
python train_streaming_ctnet.py \
  --steps 1000 \
  --batch 2 \
  --no-fineweb \
  --no-swe \
  --p-openwebmath 0.70 \
  --p-numina 0.30 \
  --out-dir runs/math_stream
```

## Auditoría del plegado reversible

```bash
python ctnet_omega_cubo6d_plegado_ctnet26.py --batch 2 --steps 1 --fp64
```

## Qué se entrena

El entrenamiento externo puede parecer normal:

```text
x -> y
loss -> backprop -> optimizer
```

Pero internamente la señal reorganiza:

```text
Z/u-p + M fija + R fija + Cubo 6D + tensor de coherencia + reserva estructural
```

El script registra:

- pérdida de tarea;
- energía de coherencia;
- omega del Cubo;
- tracking de C6;
- varianza estructural de memoria y relaciones;
- auditoría de reversibilidad;
- formas fijas de `M` y `R`.

## Archivos principales

```text
ctnet_omega_cubo6d_plegado_ctnet26.py   núcleo plegado reversible
ctnet_streaming_datasets.py             streams online sin descarga de corpus
train_streaming_ctnet.py                entrenamiento online
configs/streaming_online_default.json   configuración de referencia
scripts/train_streaming_online.sh       launcher básico
```
