# Bitácora de uso de IA

Obligatoria según la guía docente. Vale **4 puntos de E2**.

**No se puntúa usar poco.** Se puntúa el criterio: una entrada que dice
«se lo pedimos, nos dio esto, lo rechazamos porque…» vale más nota que no
haber usado ninguna herramienta.

**La línea roja:** la IA se puede usar para comprender y generar código,
citándola. **No** para el análisis automático de los resultados ni para la
discusión reflexiva sobre el trabajo. Eso es lo que evalúa la asignatura, y se
comprueba en la defensa.

---

| Fecha | Tarea | Herramienta | Qué devolvió | Qué hicimos y por qué |
|---|---|---|---|---|
| 30-sep | Comprobar que los pesos preentrenados cargan | Claude Code | `main.py` con `use_gpu = True` y rutas relativas al directorio | **Aceptado con cambios.** Daba CUDA por supuesta, pero `requirements.txt` fija `torch==2.5.1`, que en Windows es solo CPU: reventaba. Lo pasamos a `torch.cuda.is_available()` y a resolver rutas desde la raíz del repo |
| 30-sep | Instalar el entorno | Claude Code | Entorno creado desde `requirements.txt` | **Detectó un fallo nuestro.** `src/utils/utils.py` importa `timm` y no estaba declarado: en una instalación limpia el proyecto no arrancaba. Lo añadimos fijado a `timm==1.0.30` |
| 12-oct | Objetivos Neuron y Channel | Claude Code | Las dos clases, con la `x` indexando el alto | **Aceptado tras verificar.** La `x` de lucid indexa `shape[1]`, que en NHWC es el alto; en PyTorch (NCHW) eso es `act[:, z, x, y]`. Confundirlos no da error, solo optimiza la neurona traspuesta, así que añadimos un test con un mapa 3×5 para que el fallo sea detectable |
| 12-oct | Tests de los objetivos | Claude Code | Un test que comparaba dos modelos distintos | **Rechazado.** Fallaba de forma intermitente. El modelo de prueba se construía sin fijar la semilla y torch siembra su generador global con entropía del sistema, así que cada ejecución probaba un modelo distinto. Lo corregimos con `fijar_seed(0)` y comparando medias en vez de extremos |
| 13-oct | Valores de robustez a transformaciones | Claude Code | `pad 12, jitter 8, jitter_final 4`, de `standard_transforms` de lucid | **Rechazado.** Para una replicación manda el artículo, no la biblioteca: la nota al pie 9 usa `pad 16, jitter 16, jitter_final 8` y los 11 ángulos enteros de −5 a 5. Son dos fuentes que no coinciden entre sí; elegimos el artículo y lo dejamos anotado en el YAML |
| 15-oct | Primeras pruebas sobre fotos reales del test | Claude Code | 0 aciertos de 5, con las dos normalizaciones | **Rechazamos la conclusión inicial.** No era el código: las cinco especies elegidas eran de la cola larga, dos con solo 3 imágenes de entrenamiento. Filtrando a especies con 300+ imágenes, 120 fotos dan 80,0 % de top-1 |
| 15-oct | Traducir la salida del modelo a nombres de especie | Claude Code | `class_idx_to_species_id.json` derivado, porque no existe en Zenodo ni en el repo oficial | **Aceptado tras verificar.** No nos fiamos de un mapa derivado: comprobamos que los cuatro recuentos del metadata coinciden con la publicación (306 146 / 243 916 / 31 118 / 31 112 y 1081 especies) y que el top-3 de una *Alcea rosea* son tres congéneres |
| 23-oct | Regularizadores L1 y blur | Claude Code | Las fórmulas tomadas de `lucid/optvis/objectives.py` | **Aceptado con una corrección importante.** Los coeficientes del artículo (L1 −0.05, TV −0.25) **no** se pueden copiar: allí van sobre sumas de 150 528 elementos y aquí sobre medias, y además allí se maximiza y aquí se minimiza. Queda avisado en el YAML |
| 23-oct | Parametrización por píxeles | Claude Code | `ImagenPixel` | **Destapó un fallo propio.** `_comprobar_gradiente` tenía cableado `parametrizacion.espectro`, que solo existe en la versión de Fourier: la detección de unidades muertas reventaba con `AttributeError`. Lo generalizamos a todos los parámetros y lo fijamos con un test |
| 26-oct | Igualar el número de commits entre los dos | Claude Code | El procedimiento para partir y reescribir el historial | **Rechazado.** El coste no compensaba: reescribir el commit 22 de 28 cambia el SHA de los 21 siguientes, rompe el clon del compañero y deja los 5 PR apuntando a commits inexistentes. Comprobamos además que el reparto real ya estaba equilibrado por líneas (1787 frente a 1714) |
