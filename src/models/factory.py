"""Modelo de juguete del smoke test de la plantilla (run.py).

El modelo de la replicacion es el ResNet-18 preentrenado, y lo carga
construir_modelo_congelado() en src/interpretability/feature_viz.py.
"""

from __future__ import annotations

import torch.nn as nn


class ModeloPlaceholder(nn.Module):
    """Red minima para el smoke test."""

    def __init__(self, dim_entrada: int = 32, clases: int = 10):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, 64),
            nn.ReLU(),
            nn.Linear(64, clases),
        )

    def forward(self, x):
        return self.red(x)


def construir_modelo(cfg: dict) -> nn.Module:
    nombre = cfg["modelo"]["nombre"]
    if nombre == "mi_modelo":
        return ModeloPlaceholder()
    raise ValueError(f"modelo desconocido: {nombre}")
