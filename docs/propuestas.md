# Ficha de propuestas de proyecto · Hito H1

**Entrega: domingo 20 de septiembre, 23:59, en la Tarea «H1 · Propuestas de proyecto».**
**Decisión: domingo 27 de septiembre (hito H2).** Sin visto bueno no se sigue adelante.

Hasta **tres propuestas, en orden de preferencia**. La primera tiene que estar completa; la segunda y
la tercera son vuestras alternativas si la primera no pasa la puerta. Las tres tienen las mismas cuatro
comprobaciones.

---

## Cómo se entrega

1. Copiad esta página entera a un fichero `docs/propuestas.md` de vuestro repositorio y rellenadla ahí.
   Es Markdown: las tablas se editan en cualquier editor, y así queda en la historia de _commits_.
2. Marcad las casillas cambiando `[ ]` por `[x]` **solo cuando sea verdad**. Una casilla marcada sin
   evidencia en la tabla de debajo cuenta como no marcada.
3. Haced _commit_ y _push_, y en la Tarea de ALUD pegad **la URL del fichero en GitHub**. Entrega uno,
   cuenta por los dos.
4. El 27 de septiembre tendréis la decisión de cada propuesta en la retroalimentación de la Tarea. La
   aprobada se convierte en vuestro `docs/viabilidad.md`, que es el que se corrige con E1.

---

## Las dos vías

**Vía A · lista curada.** Uno de los doce papers ya comprobados, que están en el capítulo _Lista de
papers_ de este Libro: ResNet, U-Net, YOLO (v1), Vision Transformer, CycleGAN, Grad-CAM, SimCLR,
ejemplos adversarios (FGSM), Word2Vec, Transformer, BERT (solo preentrenamiento) y Neural
Collaborative Filtering. Sabemos que caben en la regla 5/20 y que tienen un aplicativo natural encima.
La ficha se rellena igual, pero es rápido. **Máximo dos parejas por paper**, por orden de entrega.

**Vía B · propuesta propia.** Cualquier paper o caso industrial que os interese, si la ficha pasa las
cuatro comprobaciones. Da más trabajo al principio, y ese trabajo de acotar es justo lo que venís a
aprender aquí.

---

## Pareja

|                                     |                                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Nombres**                         | David Fernández Fernández, Javier Santamaría Pascual                                                                    |
| **Repositorio**                     | https://github.com/javiers2004/tiiac-project-davidfernandez-javiersantamaria                          |
| **Rol profesional al que apuntáis** | Ingenieros de Machine Learning y Scrum Masters |

---

## Propuesta 1

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | Growing Neural Cellular Automata                                    |
| **Autores y año**          | Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin (2020) |
| **Enlace**                 | https://distill.pub/2020/growing-ca/ (DOI 10.23915/distill.00023)   |
| **Vía**                    | B (propuesta propia)                                                |
| **Por qué este y no otro** | Nos parece una propuesta original y visualmente atractiva. |

### Comprobación 1 · Datos

- [ ] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | No aplica. El objetivo de entrenamiento es una única imagen RGBA (p. ej. un emoji de 40×40 px) elegida libremente, no un dataset de terceros |
| **Enlace de descarga**                | No aplica (no hay dataset externo que descargar)               |
| **Tamaño**                            | Una imagen de pocos KB                                          |
| **Licencia**                          | Código y artículo bajo CC-BY 4.0                                |
| **¿Hace falta registro o solicitud?** | No                                                               |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | Sí, hay un notebook oficial de Colab (`google-research/self-organising-systems`) y repo del artículo (`distillpub/post--growing-ca`) |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | Entre el paper y el codigo se puede determinar la arquitectura completa. |
| **Qué NO especifica el paper**                           | Hiperparámetros exactos de entrenamiento (learning rate, nº de iteraciones, optimizador concreto) presentes en el codigo. |

### Comprobación 3 · Cómputo · la regla 5/20

- [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Grid de 40×40–64×64 px, ~8.3K parámetros. Deberia ser practicamente instanteneo.                     |
| **Escala que vais a hacer vosotros**           | La misma escala ya es ligera; si hiciera falta ir más rápido, grid a 32×32 y pool reducido a 512 estados          |
| **Tiempo estimado del entrenamiento completo** | Minutos en Colab gratuito (T4)                                           |
| **Qué se pierde al recortar**                  | Con un grid más pequeño el patrón objetivo pierde detalle fino; con un pool de 512 en vez de 1024 puede aparecer algo más de inestabilidad en la regeneración tras daño |
| **Hardware que vais a usar**                   | Colab gratuito                                                                                                     |

### Comprobación 4 · Aplicativo

- [ ] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | Demo interactiva de "dañar y regenerar": editor/generador de sprites o assets auto-reparables para videojuegos, o herramienta de arte generativo |
| **Quién lo usaría y para qué** | Diseñadores/as de videojuegos o artistas generativos que quieran patrones/sprites resistentes a daño |
| **Qué necesita del modelo**    | Entrada = imagen objetivo (o semilla dañada); salida = frames de regeneración paso a paso; latencia aceptable ~tiempo real (decenas de ms por paso) |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | Nada |
| **Qué haríais si pasa**                            | Nada |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 2 (alternativa)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | Feature Visualization                                               |
| **Autores y año**          | Chris Olah, Alexander Mordvintsev, Ludwig Schubert (2017)           |
| **Enlace**                 | https://distill.pub/2017/feature-visualization/ (DOI 10.23915/distill.00007) |
| **Vía**                    | B (propuesta propia)                                                |
| **Por qué este y no otro** |  Creemos que la interpretabilidad es un campo con bastante camino por recorrer y este paper es clave en la interpretabilidad de modelos de visión. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | No hace falta dataset de entrenamiento propio — se usa una red ya preentrenada (GoogLeNet/Inception sobre ImageNet) |
| **Enlace de descarga**                | Modelo preentrenado vía `tf.keras.applications` o el propio Lucid: https://github.com/tensorflow/lucid |
| **Tamaño**                            | Pesos del modelo preentrenado, del orden de decenas de MB       |
| **Licencia**                          | Apache 2.0 (TensorFlow/Lucid); artículo bajo CC-BY 4.0          |
| **¿Hace falta registro o solicitud?** | No                                                               |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | Sí — https://github.com/tensorflow/lucid, librería oficial usada para generar todas las visualizaciones del artículo |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | Sí se incluye una receta técnica detallada que consiste en: optimización en espacio de Fourier decorrelacionado + decorrelación de color (Cholesky), transformaciones estocásticas (padding 16px, jitter 16px, escalados 1/0.975/1.025/0.95/1.05, rotaciones ±5°, jitter 8px extra), Adam lr=0.05, 2560 epochs |
| **Qué NO especifica el paper**                           | Valores exactos por capa/canal para reproducir cada figura concreta del artículo; hiperparámetros finos del término de diversidad más allá de la fórmula general (Gram matrix + similitud coseno) |

### Comprobación 3 · Cómputo · la regla 5/20

- [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Optimización de una imagen (~224×224) por canal/neurona sobre GoogLeNet preentrenado, 2560 pasos de Adam          |
| **Escala que vais a hacer vosotros**           | La misma escala ya es ligera; se puede reducir a 512-1000 pasos para las primeras pruebas                         |
| **Tiempo estimado del entrenamiento completo** | Segundos a pocos minutos por imagen en GPU gratuita de Colab, factible incluso en CPU                             |
| **Qué se pierde al recortar**                  | Con menos pasos las visualizaciones son más ruidosas y menos nítidas, pero siguen siendo interpretables           |
| **Hardware que vais a usar**                   | Colab gratuito o portátil                                                                                          |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | Panel de interpretabilidad: el usuario sube una imagen o elige una capa/canal de un modelo de clasificación y obtiene la visualización generada junto con una explicación de qué ha aprendido esa parte de la red |
| **Quién lo usaría y para qué** | Equipos de ML/MLOps que necesiten auditar o explicar modelos de visión antes de ponerlos en producción |
| **Qué necesita del modelo**    | Entrada = capa/canal/clase objetivo; salida = imagen optimizada; latencia aceptable de segundos (proceso offline/batch, no tiempo real) |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | Que las visualizaciones sean difíciles de interpretar sin experiencia previa, restando valor percibido al aplicativo |
| **Qué haríais si pasa**                            | Acompañar cada visualización con ejemplos del dataset que activan la misma neurona, tal como hace el propio paper, para dar contexto |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 3 (extra — fuera de las 3 oficiales, ver nota al inicio)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | Deep attention networks reveal the rules of collective motion in zebrafish |
| **Autores y año**          | Francisco J. H. Heras, Francisco Romero-Ferrero, Robert C. Hinz, Gonzalo G. de Polavieja (2019) |
| **Enlace**                 | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007354 (DOI 10.1371/journal.pcbi.1007354) |
| **Vía**                    | B (propuesta propia)                                                |
| **Por qué este y no otro** | Nos interesan mucho las aplicaciones del Machine Learning a la biología. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | Vídeos/trayectorias de grupos de 60, 80 y 100 peces cebra (*Danio rerio*) juveniles |
| **Enlace de descarga**                | https://idtracker.ai (de los mismos autores del paper)          |
| **Tamaño**                            | 6 vídeos de 10 minutos (60 y 100 peces) + 3 vídeos de 10 minutos (80 peces), 32 fps, 20 Mpx — se puede trabajar con subconjuntos |
| **Licencia**                          | Artículo CC-BY 4.0 (open access, PLOS) |
| **¿Hace falta registro o solicitud?** | No parece necesario para el artículo |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | Sí — https://gitlab.com/polavieja_lab/fishandra, código oficial de los propios autores, libre y de código abierto |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | Sí de todos modos, con mucho detalle: pair-interaction subnetwork = MLP de 3 capas ocultas de 128 neuronas + capa de salida de 128 (ReLU), antisimetrizado respecto al eje longitudinal; aggregation subnetwork = misma estructura con salida de 1 neurona + función exponencial; loss function = cross-entropy binaria; optimizador Adam con learning rate de 1e-4 a 1e-5, batch size 500, early stopping (10 épocas sin mejora, máx. 100 épocas) |
| **Qué NO especifica el paper**                           | Tiempo de entrenamiento real ni hardware usado explícitamente (previsiblemente rápido, al ser MLPs pequeños sobre variables cinemáticas y no sobre vídeo) |

### Comprobación 3 · Cómputo · la regla 5/20

- [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | n=25 vecinos más cercanos, MLPs de 128 neuronas × 3-4 capas, entrenado sobre 9 vídeos de 10 minutos               |
| **Escala que vais a hacer vosotros**           | 1 vídeo de 10 minutos (60 peces) en vez de 9; reducir vecinos de 25 a 10; early stopping más agresivo             |
| **Tiempo estimado del entrenamiento completo** | Minutos por época en portátil o Colab gratuito — son MLPs pequeños sobre datos tabulares, no CNNs sobre vídeo |
| **Qué se pierde al recortar**                  | Con menos vecinos y menos datos, la precisión de predicción del giro bajará del 85-86% que reporta el paper, y el rango dinámico de vecinos interactuantes (8-22) estimado será menos fiable |
| **Hardware que vais a usar**                   | Portátil o Colab gratuito                                                                                          |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | Panel que, dado un vídeo o trayectorias de un grupo de animales (o robots/peatones), predice el giro/movimiento futuro de un individuo y visualiza en tiempo real las zonas de atracción/repulsión/alineación respecto a sus vecinos |
| **Quién lo usaría y para qué** | Investigadores de comportamiento colectivo, equipos de swarm robotics, o analítica de movimiento en deportes de equipo |
| **Qué necesita del modelo**    | Entrada = posición/velocidad/aceleración del individuo focal y sus n vecinos más cercanos; salida = probabilidad de giro a la derecha/izquierda; latencia aceptable de milisegundos (es un MLP pequeño) |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | Que el subconjunto de datos recortado no tenga suficientes ejemplos de giros grandes (20°-160°) para entrenar bien, ya que son los más informativos según el propio paper |
| **Qué haríais si pasa**                            | Si la accuracy cae demasiado con 1 vídeo, añadir un segundo vídeo de 60 peces antes de reducir aún más el número de vecinos |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |
