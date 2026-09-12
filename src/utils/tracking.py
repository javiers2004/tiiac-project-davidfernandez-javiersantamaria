"""Seguimiento de experimentos con Weights & Biases.

Aislado en un modulo para que el resto del codigo no dependa de wandb y para
que los tests puedan correr con modo='disabled'.
"""

from __future__ import annotations

from typing import Any


def abrir_run(cfg: dict) -> Any | None:
    """Abre un run de W&B, o devuelve None si esta desactivado."""
    wcfg = cfg.get("wandb", {})
    if wcfg.get("modo") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("aviso: wandb no esta instalado; se sigue sin registrar")
        return None

    return wandb.init(
        project=wcfg.get("proyecto", "tiiac-2026"),
        entity=wcfg.get("entidad"),
        name=cfg["experimento"]["nombre"],
        config=cfg,
        mode=wcfg.get("modo", "online"),
    )


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra metricas si hay run abierto. Silencioso si no."""
    if run is not None:
        run.log(metricas, step=paso)
