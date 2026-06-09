# CTNet: lectura dinámica de la loss

En CTNet la loss puede entrenarse con una rutina superficialmente parecida a la de un sistema diferenciable clásico:

```text
dato -> estado -> forward -> loss -> backward -> actualización
```

La diferencia no está necesariamente en la mecánica externa del entrenamiento, sino en lo que la loss está midiendo internamente.

En un Transformer autoregresivo clásico, la loss suele leerse como una métrica relativamente plana de predicción: bajar la pérdida equivale, en primera aproximación, a mejorar la predicción del siguiente token.

En CTNet, especialmente bajo el marco de masa contextual, cierre, coherencia, omega, reversibilidad y topología de carta, la loss debe leerse como una señal dinámica de relación entre la información entrante y el estado actual del sistema.

No representa sólo error. Representa interacción, cierre y coherencia de estado.

## Principio central

La coherencia CTNet se expresa como:

```text
u = p
```

a todas las escalas y desde todas las perspectivas disponibles del estado.

Aquí `u` y `p` no son dos salidas decorativas. Son dos modos o mitades de lectura de una misma estructura. Cuando `u = p`, la estructura observada cierra consigo misma: lo que aparece en una mitad de la carta coincide con lo que aparece en la otra tras la partición modal.

Por tanto, la pérdida CTNet no debe entenderse sólo como:

```text
la salida se parece al objetivo
```

sino como:

```text
el sistema logra que la información, al ser partida en modos u/p,
cierre como la misma estructura en todas sus escalas y perspectivas.
```

## Entrenamiento parecido por fuera, distinto por dentro

CTNet puede entrenarse de manera aparentemente similar a un Transformer:

```text
entrada textual -> estado -> predicción/deformación -> loss -> gradiente
```

Pero lo que ocurre internamente es de órdenes de magnitud más rico:

```text
entrada textual
-> masa contextual
-> plegado reversible
-> tensor de coherencia
-> particiones u/p
-> residuo
-> absorción
-> cierre
-> reinscripción
```

En un Transformer clásico, el objetivo se concentra principalmente en ajustar una distribución next-token.

En CTNet, incluso si existe un objetivo textual o una señal de tarea, esa señal queda envuelta por una geometría de cierre. La tarea textual puede servir como ancla de carta, pero la coherencia real viene dada por:

```text
tensor de coherencia + u/p + omega + reversibilidad + cierre
```

## Qué significa `u = p`

La condición `u = p` debe comprobarse en múltiples perspectivas:

```text
Z        estado semántico activo
M        memoria fija interna
R        relaciones internas
C6       Cubo 6D / observador de cierre
Xi       estado CTNet completo empaquetado
DeltaXi  deformación producida: Xi_out - Xi_in
```

Y en múltiples escalas:

```text
escala cruda
pooling por grupos de tokens
rotaciones/rolls de perspectiva
particiones de canal
perspectivas desplazadas
```

La idea no es que una única capa coincida consigo misma. La idea es que la estructura cierre cuando se la observa desde muchas cartas parciales.

Si `u = p` sólo en una escala, puede ser una coincidencia local. Si `u = p` aparece en muchas escalas y perspectivas, entonces hay coherencia estructural.

## Componentes de la loss

Una loss CTNet completa puede incluir componentes aparentemente familiares, pero su lectura no es la misma que en un Transformer.

```text
anchor/task      ancla de carta textual o señal de tarea
up              discrepancia u/p multiescala y multiperspectiva
coh             energía del tensor de coherencia
omega           residuo no absorbido
residual        distancia estructural observada
absorption      capacidad de absorber la distancia
closure_score   grado de cierre del régimen
rev             fidelidad reversible
structure       no colapso de memoria/relaciones fijas
speed           intensidad geométrica del tensor de coherencia
```

La pérdida textual puede bajar o subir, pero eso no agota lo que está ocurriendo. CTNet no sólo ajusta salida: reorganiza masa contextual y mide si esa reorganización cierra.

## Lectura de las componentes

### `loss_total`

Coste compuesto del estado actual. No debe leerse de forma aislada.

### `up`

Mide la discrepancia entre los modos `u` y `p` en las perspectivas observadas.

Una reducción sostenida de `up` indica que las dos mitades modales del sistema se están reconciliando.

Si `up` sube, no significa necesariamente fallo. Puede indicar que una entrada abre una obstrucción nueva o fuerza una reorganización de carta.

### `coh`

Mide tensión/coherencia interna del estado plegado mediante el tensor de coherencia.

Una bajada sostenida suele indicar estabilización geométrica, pero una subida puntual puede representar información difícil de absorber o un cambio de régimen.

### `omega`

Mide residuo no absorbido:

```text
omega = max(0, residual - absorption)
```

Si `omega = 0`, el residuo observado queda absorbido dentro del régimen actual. No significa que la tarea completa esté resuelta; significa que no hay exceso no absorbido en esa carta.

### `rev`

Mide fidelidad reversible. Debe mantenerse bajo. Si sube, la dinámica está perdiendo reversibilidad numérica o estructural.

### `anchor` / `task`

Mide la relación con una carta textual o con un objetivo externo. Es útil para mantener el estado conectado al lenguaje o a la tarea, pero no debe confundirse con toda la coherencia CTNet.

La salida textual puede ser el ancla visible; la coherencia está en el cierre global.

## Fluctuación como información

La fluctuación no es necesariamente un fallo.

En CTNet, una subida puntual puede indicar:

- aparición de una obstrucción nueva,
- cambio de régimen local,
- tensión entre la carta activa y el dato entrante,
- dificultad de absorción del residuo,
- reorganización de la masa contextual,
- cruce de una frontera topológica,
- reacomodo de memoria fija y relaciones internas,
- transición hacia un cierre más estable.

Por eso no se debe leer:

```text
loss sube = malo
loss baja = bueno
```

sino:

```text
¿qué componente sube?
¿sube up o sube sólo anchor?
¿omega se activa o sigue absorbido?
¿coh baja a largo plazo?
¿rev sigue bajo?
¿closure_score se mantiene?
¿la estructura se estabiliza después de la perturbación?
```

Una curva sana puede no ser monótona. Puede presentar:

```text
subida -> tensión de carta
meseta -> reorganización interna
bajada -> absorción
nuevo pico -> nueva obstrucción
estabilización -> cierre de régimen
```

## Masa contextual y salida

En CTNet, la salida no debe entenderse únicamente como una generación autoregresiva de tokens. Debe entenderse como la reinscripción más coherente de una masa contextual ya formada.

La forma conceptual es:

```text
prompt
-> deformación de la masa contextual
-> estabilización de régimen
-> cierre u/p multiescala
-> reinscripción visible en carta textual
```

Por tanto, una loss fiel a CTNet no debe limitarse a:

```text
out.z ≈ target_z
```

Esa métrica puede servir como ancla de carta, pero no como filosofía completa.

El criterio más fiel es:

```text
la información aprendida debe cerrar como u = p
bajo el tensor de coherencia
a todas las escalas y perspectivas disponibles.
```

## Sobre negativos y ranking de candidatos

Las pérdidas de ranking, margen o candidatos negativos pueden ser herramientas auxiliares de diagnóstico, pero no son el núcleo de CTNet.

Pueden ayudar a comprobar si una salida pertenece mejor que otra a una masa contextual, pero si se vuelven el criterio principal, se corre el riesgo de volver a una métrica externa de parecido o separación.

El núcleo sigue siendo:

```text
coherencia = u = p
```

El ranking de candidatos sólo tiene sentido si se evalúa como una consecuencia del cierre, no como reemplazo del cierre.

## Reglas prácticas de diagnóstico

Durante entrenamiento CTNet, mirar la loss total no basta. Deben revisarse las relaciones:

```text
up baja a medio/largo plazo
coh baja o se estabiliza tras perturbaciones
omega permanece bajo o vuelve a cero
rev permanece cerca de cero
closure_score se mantiene alto
anchor mantiene conexión con la carta textual
M y R no colapsan
Xi y DeltaXi mantienen u/p bajo
```

Casos típicos:

### Caso A: `coh` baja, `omega=0`, `rev` bajo, pero `up` alto

El sistema está absorbiendo residuo y conserva reversibilidad, pero todavía no ha reconciliado sus modos internos. La geometría puede estar estable, pero la coherencia modal sigue incompleta.

### Caso B: `up` baja, `coh` baja, `omega=0`, `rev` bajo

La masa contextual empieza a cerrar de forma fuerte. Es una señal estructuralmente buena.

### Caso C: `omega` sube de forma persistente

El residuo supera la absorción. Puede indicar que la carta actual no puede contener la información entrante o que el régimen necesita otra parametrización.

### Caso D: `rev` sube

La dinámica está perdiendo fidelidad reversible. Es una señal estructuralmente grave.

### Caso E: `anchor` baja pero `up` no baja

El sistema puede estar ajustando la carta textual de forma superficial, pero sin cierre modal profundo.

### Caso F: `up` baja pero `anchor` no baja

El sistema puede estar logrando coherencia interna, pero todavía no se reinscribe bien en la carta textual observada.

## Resumen

CTNet puede entrenarse con una mecánica exterior parecida a otros sistemas diferenciables, incluso a un Transformer en el sentido de forward/loss/backward.

Pero la interpretación de lo que ocurre es diferente.

En CTNet, la loss es un registro de asimilación y cierre, no sólo una métrica plana de error.

Su forma expresa el acoplamiento entre:

```text
información entrante
masa contextual
carta activa
topología interna
partición u/p
tensor de coherencia
residuo
absorción
cierre
reversibilidad
reinscripción de salida
```

Por eso la lectura correcta no es buscar una curva lisa que baje siempre, sino observar cómo el sistema atraviesa obstrucciones, absorbe residuo, conserva reversibilidad y reduce la discrepancia `u/p` en todas las escalas y perspectivas.
