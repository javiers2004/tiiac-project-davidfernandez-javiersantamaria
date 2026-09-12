"""Capa de servicio: el APLICATIVO que va encima de la replicacion.

Vale 8 puntos de E2 y es lo que convierte una replicacion en un producto.

Su unica obligacion arquitectonica: cargar un checkpoint y responder. NO
entrena, NO conoce el dataset de entrenamiento, NO importa nada de
src/training/. Esa separacion es el contrato del checkpoint, y es lo que hace
que el aplicativo se pueda desplegar sin arrastrar el codigo de entrenamiento.

    python -m src.app.serve --checkpoint checkpoints/mejor.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.factory import construir_modelo


class Predictor:
    def __init__(self, checkpoint: Path, cfg: dict):
        self.modelo = construir_modelo(cfg)
        estado = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.modelo.load_state_dict(estado)
        self.modelo.eval()

    @torch.no_grad()
    def predecir(self, x: torch.Tensor) -> torch.Tensor:
        return self.modelo(x).softmax(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    print(f"cargando {args.checkpoint} ...")
    # Sustituye por tu interfaz: CLI interactiva, FastAPI, Gradio o Streamlit.
    print("TODO: aqui va la interfaz del aplicativo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
