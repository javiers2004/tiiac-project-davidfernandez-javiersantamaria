## Qué hace este PR

<!-- Una o dos frases. Si necesitas un párrafo, quizá son dos PRs. -->

## Cómo se ha comprobado

<!-- Comando ejecutado, test añadido, run de W&B. -->

---

## Revisión (la rellena quien NO escribió el código)

### Correctitud numérica
- [ ] Formas de tensores comprobadas en las fronteras del pipeline
- [ ] Ejes correctos en `softmax`, medias y `argmax` (`dim=1` para clases)
- [ ] `.detach()` / `no_grad()` donde toca; no se acumula grafo en validación

### Fugas de datos
- [ ] La normalización usa estadísticas de **train**, no del conjunto completo
- [ ] El split se hace antes de cualquier transformación ajustada a los datos
- [ ] No hay *augmentation* aplicada a validación ni a test

### Reproducibilidad
- [ ] Las *seeds* se fijan en un único sitio
- [ ] Ningún hiperparámetro escrito a fuego: todos en `configs/`
- [ ] Sin rutas absolutas
- [ ] El *run* queda registrado en W&B

### Legibilidad
- [ ] Sin números mágicos sin explicar
- [ ] Sin código duplicado
- [ ] Funciones de tamaño razonable
