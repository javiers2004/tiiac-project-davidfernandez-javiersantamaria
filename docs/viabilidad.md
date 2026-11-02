# Viabilidad

Estado de la propuesta aprobada, comprobado sobre lo que hay en el repositorio.
La propuesta original está en [`propuestas.md`](propuestas.md) (Propuesta 2). La
decisión y las condiciones del profesor están en la retroalimentación de la Tarea
del hito H2.

| | |
|---|---|
| **Paper** | Olah, Mordvintsev y Schubert (2017), *Feature Visualization* |
| **Enlace** | https://distill.pub/2017/feature-visualization/ (DOI 10.23915/distill.00007) |
| **Vía** | B (propuesta propia) |

## Qué cambió respecto a la propuesta

| | Propuesta | Proyecto |
|---|---|---|
| Modelo | GoogLeNet/Inception preentrenado | ResNet-18 preentrenado |
| Dataset del modelo | ImageNet | Pl@ntNet-300K |
| Biblioteca | TensorFlow / lucid | PyTorch; lucid queda como referencia |

Los pesos siguen congelados: lo que se optimiza es la imagen de entrada.

## Comprobación 1 · Datos

- [x] Son públicos y descargables hoy.
- [x] La licencia permite el uso académico.

| | |
|---|---|
| **Dataset** | Pl@ntNet-300K: 306 146 imágenes de 1081 especies |
| **Descarga** | `python scripts/get_data.py`, desde [Zenodo, record 5645731](https://zenodo.org/records/5645731) |
| **Pesos** | `python scripts/get_data.py --pesos`, del enlace de [plantnet/PlantNet-300K](https://github.com/plantnet/PlantNet-300K) |
| **Tamaño** | Dataset 31,7 GB, no necesario para las visualizaciones. Pesos 94 MB, metadatos 70 MB |
| **Licencia** | CC BY 4.0, según la ficha de Zenodo |
| **¿Registro?** | No |

## Comprobación 2 · Especificación

- [x] Hay implementación de referencia.

| | |
|---|---|
| **Código de referencia** | [tensorflow/lucid](https://github.com/tensorflow/lucid) |
| **Qué no especifica el paper** | Lo que se decidió en cada caso está en la tabla «Decisiones donde el paper no especifica» del [README](../README.md) |

## Comprobación 3 · Cómputo · regla 5/20

- [x] Cabe en el presupuesto, con el recorte escrito.

| | |
|---|---|
| **Escala del paper** | 2560 pasos de Adam por imagen, a unos 224×224 |
| **Escala del proyecto** | 256 pasos por defecto, a 224×224 |
| **Tiempo** | Una imagen de 256 pasos: unos 18 s en CPU. No hay entrenamiento |
| **Smoke test** | `python run.py --smoke`: unos 2 s. Solo ejercita el modelo de juguete de la plantilla |
| **Qué se pierde** | Las imágenes salen menos refinadas. Subir `optimizacion.pasos` a 2560 recupera la escala del paper |

## Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio encima de la replicación.

| | |
|---|---|
| **Qué se construye encima** | Una herramienta de inspección en Streamlit, en `src/app/serve.py`: subes una foto y ves qué especie predice el modelo, qué neuronas se activan, dónde se activan y qué patrón busca cada una |
| **Qué necesita del modelo** | Solo el checkpoint, a través de `src/interpretability/api.py` |
| **Latencia** | Medir una foto es inmediato; generar un patrón tarda de segundos a minutos según los pasos, por eso va detrás de un botón |

## Riesgo principal

| | |
|---|---|
| **Riesgo de la propuesta** | Que las visualizaciones sean difíciles de interpretar sin experiencia previa |
| **Respuesta prevista** | Acompañarlas de fotos reales que activan la misma neurona. Implementado en `ejemplos_del_dataset()` |
