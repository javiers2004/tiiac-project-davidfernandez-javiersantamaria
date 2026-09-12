#!/usr/bin/env python3
"""
Punto de entrada unico del proyecto.

    python run.py --smoke                        # pipeline entero en miniatura, < 5 min en CPU
    python run.py --config configs/default.yaml  # entrenamiento completo, < 20 min en T4
    python run.py --config configs/default.yaml --solo-evaluar

Este fichero es deliberadamente corto: orquesta, no implementa. La logica vive
en src/. Si run.py crece por encima de un centenar de lineas, algo tendria que
haber bajado a un modulo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import fijar_seed          # noqa: E402
from src.utils.tracking import abrir_run       # noqa: E402
from src.data.loaders import construir_loaders  # noqa: E402
from src.models.factory import construir_modelo  # noqa: E402
from src.training.loop import entrenar, evaluar  # noqa: E402


def cargar_config(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="fichero de configuracion")
    ap.add_argument("--smoke", action="store_true",
                    help="usa configs/smoke.yaml: el pipeline entero en miniatura")
    ap.add_argument("--solo-evaluar", action="store_true",
                    help="carga el checkpoint y evalua, sin entrenar")
    args = ap.parse_args()

    ruta = Path("configs/smoke.yaml") if args.smoke else args.config
    if not ruta.is_file():
        print(f"error: no existe {ruta}", file=sys.stderr)
        return 2
    cfg = cargar_config(ruta)

    # 1 · reproducibilidad, antes que nada
    fijar_seed(cfg["experimento"]["seed"])

    # 2 · seguimiento del experimento
    run = abrir_run(cfg)

    # 3 · datos
    train_loader, val_loader = construir_loaders(cfg)

    # 4 · modelo
    modelo = construir_modelo(cfg)

    # 5 · entrenar y evaluar
    if not args.solo_evaluar:
        entrenar(modelo, train_loader, val_loader, cfg, run)
    metricas = evaluar(modelo, val_loader, cfg)

    # 6 · el smoke test TIENE que imprimir una metrica: es la prueba de que el
    #     pipeline llega hasta el final. El checker lo comprueba.
    print("\n=== metricas ===")
    for k, v in metricas.items():
        print(f"{k}: {v:.4f}")

    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
