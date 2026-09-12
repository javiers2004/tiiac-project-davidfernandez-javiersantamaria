"""Construccion del modelo.

Aqui va tu replicacion. El objetivo de E1 es que el forward produzca las
FORMAS que declara el paper, aunque los pesos sean aleatorios.
"""

from __future__ import annotations

import torch.nn as nn


class ModeloPlaceholder(nn.Module):
    """Sustituye esto por la arquitectura del paper."""

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
