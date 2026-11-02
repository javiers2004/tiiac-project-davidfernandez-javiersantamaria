#!/usr/bin/env python3
"""Acierto del clasificador sobre las fotos de data/muestras.

    python scripts/get_data.py --solo-metadatos
    python scripts/get_data.py --fotos 120
    python scripts/evaluar_fotos.py

No es una metrica del articulo, que no publica ninguna. Es la comprobacion de
que el checkpoint y el preprocesado estan bien: si la normalizacion o el mapa de
clases estuvieran mal, el acierto se hundiria sin dar ningun error.

Cada foto se llama <id de especie>_<hash>.jpg, asi que la etiqueta sale del
nombre del fichero. Con WANDB_API_KEY el resultado queda registrado en W&B.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.interpretability.api import (  # noqa: E402
    CONFIG,
    cargar_especies,
    cargar_modelo,
    predecir,
)
from src.interpretability.feature_viz import cargar_config  # noqa: E402
from src.utils.tracking import abrir_run, registrar  # noqa: E402

TOP_K = 5


def main() -> int:
    """Calcula top-1 y top-5 sobre data/muestras e imprime el resultado."""
    especies = cargar_especies()
    fotos = sorted((RAIZ / "data" / "muestras").glob("*.jpg"))
    if not especies or not fotos:
        print("error: faltan datos. Ejecuta scripts/get_data.py --solo-metadatos "
              "y --fotos 120", file=sys.stderr)
        return 2

    cfg = cargar_config(CONFIG)
    modelo = cargar_modelo(cfg)
    clase_de = {sid: clase for clase, (sid, _) in especies.items()}

    top1 = top5 = 0
    for foto in fotos:
        clase = clase_de[foto.name.split("_")[0]]
        apuestas = [p.clase for p in predecir(foto, top_k=TOP_K, modelo=modelo, cfg=cfg)]
        top1 += apuestas[0] == clase
        top5 += clase in apuestas

    n = len(fotos)
    print(f"{n} fotos: top-1 {top1 / n:.1%}  top-5 {top5 / n:.1%}")

    run = abrir_run(cfg, "evaluacion-fotos-test")
    registrar(run, {"test/top1": top1 / n, "test/top5": top5 / n, "test/fotos": n})
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
