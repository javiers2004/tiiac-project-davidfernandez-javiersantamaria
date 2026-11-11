# Visualización de características sobre Pl@ntNet-300K

Replicación de **Olah, Mordvintsev & Schubert, 2017 · *Feature Visualization***
sobre el ResNet-18 preentrenado en **Pl@ntNet-300K**, con una herramienta de
inspección construida encima: subes una foto de una planta, te dice qué
neuronas activa y te genera el patrón que cada una busca.

- **Paper:** [distill.pub/2017/feature-visualization](https://distill.pub/2017/feature-visualization/) · DOI [10.23915/distill.00007](https://doi.org/10.23915/distill.00007)
- **Implementación de referencia:** [tensorflow/lucid](https://github.com/tensorflow/lucid) (TensorFlow; aquí se reimplementa en PyTorch y se contrasta contra ella)
- **Dataset y pesos:** [Pl@ntNet-300K](https://zenodo.org/records/5645731) · [plantnet/PlantNet-300K](https://github.com/plantnet/PlantNet-300K)
- **Seguimiento de experimentos:** https://wandb.ai/javier-santamaria-p-deusto/tiiac-2026
- **Autoría:** Javier Santamaría Pascual, David Fernández Fernández

## Resultado

**Este artículo no publica ninguna métrica.** *Feature Visualization* es un
trabajo cualitativo: su resultado son figuras, no cifras. No hay un número que
igualar, así que lo que se reproduce son sus **técnicas** y sus **figuras**.

| Técnica del artículo | Reproducida | Evidencia |
|---|---|---|
| Objetivos: Neuron, Channel, Layer/DeepDream, Class Logits, Class Probability | sí | `src/interpretability/objetivos.py` |
| Direction, suma de neuronas, interpolación | sí | sección «Interaction between Neurons» |
| Diversity | sí | `generar_diversas()` |
| Preconditioning: espacio de Fourier con color decorrelado | sí | contrastado línea a línea con `lucid/optvis/param/` |
| Transformation robustness | sí | valores de la nota al pie 9 del artículo |
| Penalización de frecuencia: L1, variación total, desenfoque | sí | fórmulas tomadas de `lucid/optvis/objectives.py` |
| Dataset examples | sí | `ejemplos_del_dataset()` |
| Learned priors (GAN/VAE) | **no** | fuera de alcance: requiere entrenar un modelo generativo |

Como comprobación cuantitativa de que el checkpoint y el preprocesado son
correctos —no como métrica del artículo— el clasificador acierta **80,0 % de
top-1 y 95,0 % de top-5** sobre 120 fotos del split de test, una por especie,
entre las 144 especies con 300 o más imágenes de entrenamiento. Se reproduce con
`python scripts/get_data.py --fotos 120` y `python scripts/evaluar_fotos.py`.

La reproducción del argumento central del artículo es medible: desactivando la
robustez a transformaciones (`--sin-robustez`), la activación final **sube** de
+4,93 a +7,15 mientras la aspereza de la imagen empeora de 0,164 a 0,245 (canal
181 de `layer3.1.conv2`, en CPU). Mejor óptimo, peor imagen, que es exactamente lo
que el artículo defiende. Las dos cifras se imprimen al terminar cada ejecución.

## Instalación

```bash
git clone https://github.com/javiers2004/tiiac-project-davidfernandez-javiersantamaria
cd tiiac-project-davidfernandez-javiersantamaria
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Después, los pesos preentrenados. Sin ellos no funciona nada salvo los tests y el
smoke test:

```bash
python scripts/get_data.py --pesos            #  94 MB · a models/, con MD5 verificado
```

Vienen del [repositorio oficial de Pl@ntNet-300K](https://github.com/plantnet/PlantNet-300K)
y no se versionan.

## Datos

```bash
python scripts/get_data.py --solo-metadatos   #  70 MB · nombres de especie
python scripts/get_data.py --fotos 20         # ~2 MB · fotos de prueba del test
python scripts/get_data.py                    # 31,7 GB · el dataset entero
```

**Para reproducir las visualizaciones no hace falta el dataset.** Los dos
primeros modos extraen solo lo necesario del zip remoto mediante peticiones
HTTP por rango: un zip guarda su índice al final, así que se leen unos 60 MB de
índice y se bajan únicamente los miembros que interesan.

Origen: [Zenodo, record 5645731](https://zenodo.org/records/5645731), versión
1.1. 306 146 imágenes de 1081 especies. MD5 verificado por el script.
Licencia del dataset: CC BY 4.0 (ver la ficha de Zenodo).

## Demostración

Con los pesos descargados, y mejor con los metadatos y unas fotos, que dan los
nombres de especie y las muestras:

```bash
streamlit run src/app/serve.py    # la interfaz web
python scripts/demo.py            # el mismo recorrido, por terminal
python scripts/sesion.py          # paso a paso, enseñando cada llamada a la API
```

Subes una foto o eliges una muestra, y ves qué especie predice el modelo, qué
neuronas se activan, dónde se activan y qué patrón busca cada una. Los detalles
están en [`docs/informe-tecnico-v2.pdf`](docs/informe-tecnico-v2.pdf).

## Ejecución

```bash
# Una visualización: qué patrón busca el canal 181 de layer3.1.conv2
python -m src.interpretability.feature_viz --tipo canal --capa layer3.1.conv2 --canal 181

# El logit de una clase concreta (Alcea rosea)
python -m src.interpretability.feature_viz --tipo logit --capa fc --canal 123

# DeepDream de una capa entera
python -m src.interpretability.feature_viz --tipo capa --capa layer3

# Diversidad: varias versiones distintas de la misma neurona
python -m src.interpretability.feature_viz --tipo canal --capa layer4 --canal 443 --lote 4 --diversidad

# Interpolación entre dos neuronas
python -m src.interpretability.feature_viz --tipo interpolacion --capa layer4 --canales 463 233 --lote 6

# Todos los mandos de las figuras del artículo
python -m src.interpretability.feature_viz --help

# Smoke test del esqueleto de la plantilla (ver «Reproducibilidad»)
python run.py --smoke
```

Las imágenes salen en `outputs/feature_viz/`. **No hace falta cuenta de W&B:** sin
clave se avisa y se sigue sin registrar. Para registrar la ejecución, con la clave
en el entorno (en una cuenta sin acceso al proyecto, cambia `wandb.entidad` en
`configs/feature_viz.yaml`):

```bash
WANDB_API_KEY=... python -m src.interpretability.feature_viz --canal 181
```

Desde Python, la fachada que usa el aplicativo:

```python
from src.interpretability.api import cargar_modelo, listar_neuronas, generar_imagen

m = cargar_modelo()
top = listar_neuronas("mi_planta.jpg", capas="layer3.1.conv2", top_k=5, modelo=m)
img = generar_imagen(top[0], modelo=m)
```

## Estructura

```
configs/                    Todos los hiperparámetros. Ninguno escrito a fuego
src/interpretability/
  objetivos.py              Qué se maximiza: los objetivos del artículo
  feature_viz.py            El bucle, la parametrización y la robustez
  api.py                    La fachada que consume el aplicativo
src/utils/                  Semillas, seguimiento de experimentos, utilidades
src/app/serve.py            El aplicativo, en Streamlit
scripts/get_data.py         Descarga de pesos, metadatos, fotos y dataset
scripts/evaluar_fotos.py    El acierto del clasificador que cita este README
scripts/demo.py, sesion.py  La demostración por terminal
docs/                       Viabilidad, propuestas y guía de la demostración
tests/                      133 tests de forma, invariancia y reproducibilidad
run.py                      Punto de entrada de la plantilla (ver abajo)
```

## Reproducibilidad

- Semilla fijada en `src/utils/seed.py`, valor en `configs/`. En CPU, entre procesos
  distintos, la activación puede variar en torno a un 1 %.
- Dependencias con versión exacta en `requirements.txt`.
- **Presupuesto de cómputo:** una visualización de 256 pasos a 224×224 tarda
  **18 s en CPU** (torch 2.5.1, el entorno fijado) y **3,1 s en una GTX 1650
  Ti**. El barrido más grande de la memoria, 2142 optimizaciones, tardó
  118 minutos en GPU.
- **Qué se recortó respecto al artículo:** sus figuras usan **2560 pasos** y
  aquí el valor por defecto son **256**, porque 2560 son 2,9 minutos por imagen
  en CPU y un barrido de 16 canales pasaría de 5 a 45 minutos. Las imágenes
  salen menos refinadas; subir `optimizacion.pasos` a 2560 reproduce la
  fidelidad del artículo sin tocar código.
- **`run.py` no ejecuta este proyecto.** Es el punto de entrada que venía en la
  plantilla de la asignatura y sigue entrenando su modelo de juguete sobre
  datos sintéticos. Como aquí no se entrena nada —los pesos van congelados y lo
  que se optimiza es la imagen— nunca se reconectó. El trabajo real se lanza
  con `python -m src.interpretability.feature_viz`.

## Decisiones donde el paper no especifica

| Ambigüedad | Qué decidimos | Por qué |
|---|---|---|
| Qué normalización usó el checkpoint | Las de ImageNet | Lo medimos sobre 120 fotos del test: ImageNet da 80,0 % de top-1 y Pl@ntNet-300K 79,2 %, con el top-5 idéntico. La diferencia es de una imagen en 120, así que el checkpoint es insensible a esto |
| El artículo y lucid dan valores de robustez distintos | Los del artículo (`pad 16`, `jitter 16`, `jitter_final 8`, 11 ángulos de −5 a 5) | Para una replicación manda el paper, no la biblioteca. `standard_transforms` de lucid usa `pad 12, jitter 8, jitter_final 4` y otro rango de escalas |
| Los coeficientes de las figuras (L1 −0.05, TV −0.25, Blur −1) | No se copian literalmente | Allí van sobre **sumas** de 150 528 elementos y aquí los términos se **promedian**; hay un factor de ~150 000 entre ambas escalas. Y allí se maximiza y se resta, aquí se minimiza y se suma |
| El mapa clase → especie no existe | Derivado de cómo `ImageFolder` ordena los nombres de carpeta | El fichero `class_idx_to_species_id.json` está documentado en el repo oficial pero no está ni en el zip de Zenodo, ni en el repositorio, ni en sus releases. Verificado: los cuatro recuentos del metadata coinciden con la publicación, y el top-3 de una *Alcea rosea* son tres congéneres |
| El orden de ejes al pasar de lucid a PyTorch | `act[:, z, x, y]`, con `x` indexando el alto | La `x` de lucid indexa `shape[1]`, que en NHWC es el **alto**. Confundirlos no da error: optimiza la neurona traspuesta. Hay un test con un mapa 3×5 para que el fallo sea detectable |
| Qué capas ofrecer al usuario | Las convoluciones pre-ReLU, no las salidas de bloque | Las salidas de bloque son post-ReLU y arrancan en cero en el 56 % de las posiciones centrales de `layer4`. Gradiente cero significa devolver el ruido inicial sin avisar; ahora salta la excepción `UnidadMuerta` |

## Tests

```bash
pytest tests/ -q
```

133 tests. No dependen de los `.tar`: usan un ResNet-18 sin entrenar, porque
los pesos están en `.gitignore` y en integración continua no existen.

## Uso de IA

Documentado en [`AI_USAGE.md`](AI_USAGE.md), según la guía docente.
