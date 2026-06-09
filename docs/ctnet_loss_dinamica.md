# CTNet: lectura dinámica de la loss

En CTNet la loss no debe interpretarse igual que en un Transformer autoregresivo clásico.

En un Transformer, la loss suele leerse como una métrica relativamente plana de predicción: bajar la pérdida equivale, en primera aproximación, a mejorar la predicción del siguiente token. Una subida puntual suele interpretarse como ruido, lote difícil, inestabilidad o tasa de aprendizaje inadecuada.

En CTNet, especialmente bajo el marco de masa contextual, cierre, coherencia, omega, reversibilidad y topología de carta, la loss debe leerse como una señal dinámica de relación entre la información entrante y el estado actual del sistema.

No representa sólo error. Representa interacción.

## Principio

La loss en CTNet registra cómo el sistema asimila una entrada en un momento concreto de su evolución.

Una fluctuación de la loss puede indicar:

- aparición de una obstrucción nueva,
- cambio de régimen local,
- tensión entre la carta activa y el dato entrante,
- dificultad de absorción del residuo,
- reorganización de la masa contextual,
- cruce de una frontera topológica,
- reacomodo de memoria fija y relaciones internas,
- transición hacia un cierre más estable.

Por eso, en CTNet no se debe leer:

```text
loss sube = malo
loss baja = bueno
```

sino:

```text
¿qué componente sube?
¿omega se activa o sigue absorbido?
¿coh baja a largo plazo?
¿rev sigue bajo?
¿margin mejora?
¿closure_score se mantiene?
¿la estructura se estabiliza después de la perturbación?
```

## Diferencia con Transformer

### Transformer

En un Transformer clásico:

```text
entrada + contexto externo/KV-cache -> distribución next-token -> loss token-level
```

La pérdida mide principalmente divergencia entre la distribución predicha y el token esperado.

### CTNet

En CTNet:

```text
entrada -> deformación de masa contextual -> estado cerrado -> reinscripción más coherente
```

La pérdida no debe medir únicamente si el vector producido se parece a un objetivo. Debe medir si la continuación pertenece mejor al estado contextual que sus alternativas falsas.

Por eso una loss CTNet completa debe involucrar señales como:

```text
coh             tensión/coherencia interna
omega           residuo no absorbido
residual        distancia estructural observada
absorption      capacidad de absorber la distancia
closure_score   cierre del régimen
rev             fidelidad reversible
margin          pertenencia de la salida verdadera frente a falsas
ok / okM        si la reinscripción correcta gana
speed           intensidad geométrica del tensor de coherencia
```

## Lectura de las componentes

### `loss_total`

Coste compuesto del estado actual. No debe leerse de forma aislada.

### `coh`

Mide tensión/coherencia interna del estado plegado. Una bajada sostenida suele indicar estabilización geométrica.

### `omega`

Mide residuo no absorbido:

```text
omega = max(0, residual - absorption)
```

Si `omega = 0`, el residuo observado queda absorbido dentro del régimen actual. No significa que la tarea completa esté resuelta; significa que no hay exceso no absorbido en esa carta.

### `rev`

Mide fidelidad reversible. Debe mantenerse bajo. Si sube, la dinámica está perdiendo reversibilidad numérica o estructural.

### `margin`

Mide si la continuación verdadera pertenece mejor que las negativas:

```text
margin = negMinEnergy - trueEnergy
```

Si `margin > 0`, la continuación verdadera tiene menor energía que la mejor negativa.

Si `margin < 0`, alguna reinscripción falsa cierra mejor que la verdadera bajo la métrica actual.

### `ok`

Indica si la verdadera gana sin exigir margen adicional.

### `okM`

Indica si la verdadera gana con margen fuerte.

```text
ok  = trueEnergy < negMinEnergy
okM = trueEnergy + required_margin < negMinEnergy
```

## Fluctuación como información

La fluctuación no es necesariamente un fallo.

En CTNet, una subida puntual puede ser un signo de que el sistema está encontrando una estructura que no pertenece todavía a su cierre actual. Esa subida es una obstrucción visible. Si después `coh` baja, `rev` se conserva y `omega` vuelve a cero, el sistema ha absorbido la perturbación.

La pregunta importante es si la perturbación deja una deformación útil o sólo introduce ruido.

Una curva sana puede no ser monótona. Puede presentar:

```text
subida -> tensión de carta
meseta -> reorganización interna
bajada -> absorción
nuevo pico -> nueva obstrucción
estabilización -> cierre de régimen
```

## Masa contextual y salida

En CTNet, la salida no debe entenderse como una generación autoregresiva de tokens. Debe entenderse como la reinscripción más coherente de una masa contextual ya formada.

La forma conceptual es:

```text
prompt
-> deformación de la masa contextual
-> estabilización de régimen
-> evaluación de posibles reinscripciones
-> salida como forma textual de menor residuo y mayor cierre
```

Por tanto, una loss fiel a CTNet no debe limitarse a:

```text
out.z ≈ target_z
```

Esa métrica puede servir como ancla de carta, pero no como filosofía completa.

La métrica más fiel es:

```text
la continuación verdadera debe cerrar mejor que continuaciones falsas
```

o, de forma operacional:

```text
trueEnergy + margin < everyNegativeEnergy
```

## Reglas prácticas de diagnóstico

Durante entrenamiento CTNet, mirar la loss total no basta. Deben revisarse las relaciones:

```text
coh baja a medio/largo plazo
omega permanece bajo o vuelve a cero tras perturbaciones
rev permanece cerca de cero
margin tiende a positivo
ok aumenta
okM aumenta
closure_score se mantiene alto
```

Casos típicos:

### Caso A: `coh` baja, `omega=0`, `rev` bajo, `margin<0`

El sistema está cerrando internamente, pero todavía no distingue bien la reinscripción verdadera de las falsas. La geometría se estabiliza, pero la pertenencia semántica sigue débil.

### Caso B: `coh` baja, `margin>0`, `ok=1`, `rev` bajo

La masa contextual empieza a seleccionar la salida verdadera como forma de cierre.

### Caso C: `omega` sube de forma persistente

El residuo supera la absorción. Puede indicar que la carta actual no puede contener la información entrante o que el régimen necesita otra parametrización.

### Caso D: `rev` sube

La dinámica está perdiendo fidelidad reversible. Es una señal estructuralmente grave.

## Resumen

En CTNet, la loss es un registro de asimilación, no sólo una métrica de error.

Su forma expresa el acoplamiento entre:

```text
información entrante
masa contextual
carta activa
topología interna
residuo
absorción
cierre
reversibilidad
reinscripción de salida
```

Por eso la lectura correcta no es buscar una curva lisa que baje siempre, sino observar cómo el sistema atraviesa obstrucciones, absorbe residuo, conserva reversibilidad y mejora la pertenencia de la salida verdadera dentro de su propio régimen contextual.
