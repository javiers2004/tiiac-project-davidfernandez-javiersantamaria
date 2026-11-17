# Contexto del proyecto

<!--
Fichero de contexto para asistentes de IA (Claude Code, Copilot, Cursor…).
Se explica en la sesión A1 del 9 de noviembre.

Dos criterios para escribirlo:
  1. Va lo que le dirías a alguien que se incorpora hoy al proyecto.
  2. Si te encuentras repitiendo la misma corrección tres veces, esa
     corrección va aquí.

Dos páginas se leen enteras. Quince, no.
-->

## Qué es esto

Replicación de Olah, Mordvintsev y Schubert (2017), *Feature Visualization*, sobre
un ResNet-18 preentrenado en Pl@ntNet-300K. El artículo no publica métricas: el
objetivo es reproducir sus técnicas y sus figuras. Encima se construye una
interfaz web de inspección (`src/app/serve.py`, sin framework: biblioteca
estándar y una página con Preact incluido).

Los pesos no se entrenan. Lo que se optimiza es la imagen de entrada.

## Restricciones del curso

- **Regla 5/20:** `python run.py --smoke` en menos de 5 min en CPU;
  el entrenamiento completo en menos de 20 min en una T4 de Colab gratuito.
- Todo entrenamiento se registra en Weights & Biases.
- Las semillas se fijan **solo** en `src/utils/seed.py`.
- Los hiperparámetros viven **solo** en `configs/`. Ninguno escrito a fuego.

## Cómo se ejecuta

```bash
python scripts/get_data.py --pesos todos       # los checkpoints (ResNet-18, 50 y 152)
python -m src.interpretability.feature_viz --capa layer3.1.conv2 --canal 181
python -m src.app.serve                        # el aplicativo, en http://127.0.0.1:8501
python run.py --smoke                          # smoke test de la plantilla (modelo de juguete)
pytest tests/ -q                               # tests
python tools/tiiac_check.py .                  # comprobador de reproducibilidad
```

## Convenciones

- Sin rutas absolutas, nunca.
- Toda función pública lleva docstring y anotaciones de tipo.
- **La normalización usa estadísticas de entrenamiento, no del conjunto completo.**
  Lo contrario es una fuga de datos: no da error e infla la métrica.
- El `data augmentation` se aplica a train, jamás a validación ni a test.
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga un
  checkpoint y responde.

## Lo que NO hay que tocar

- Los splits: son los oficiales de Pl@ntNet-300K.

## Estado actual

Actualiza esto cuando cambie. Es lo que evita que el asistente reinvente piezas
que ya existen.

- Hecho: los objetivos, la interacción entre neuronas, la diversidad, la
  regularización, el preconditioning y el aplicativo. La tabla está en el README.
- El aplicativo es la interfaz diseñada en Claude Design ("Inspector de
  neuronas", v13) portada tal cual a `src/app/static/`. La lógica de la página
  está en `app.js`; las reglas (qué capa es cada bloque, qué objetivo sale de
  cada selección, cómo se traducen los mandos al YAML) en `src/app/servicio.py`.
  Rangos de los mandos y checkpoints de cada modelo en `configs/app.yaml`.
- Fuera de alcance: learned priors (GAN/VAE).
