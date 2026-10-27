"""Seguimiento de experimentos con Weights & Biases.

Aislado en un modulo para que el resto del codigo no dependa de wandb y para
que los tests puedan correr con modo='disabled'.
"""

from __future__ import annotations

from typing import Any


def abrir_run(cfg: dict, nombre: str | None = None) -> Any | None:
    """Abre un run de W&B, o devuelve None si esta desactivado.

    Args:
        cfg: la configuracion completa, que se guarda con el run.
        nombre: como se llama el run en la web. Si es None se usa
            experimento.nombre, que es el mismo para todos y hace que una
            lista de veinte runs sea ilegible.
    """
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
        name=nombre or cfg["experimento"]["nombre"],
        config=cfg,
        mode=wcfg.get("modo", "online"),
    )


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra metricas si hay run abierto. Silencioso si no."""
    if run is not None:
        run.log(metricas, step=paso)


def registrar_imagen(run: Any | None, clave: str, imagen, paso: int | None = None,
                     titulo: str = "") -> None:
    """Sube una imagen al run, si hay run y si wandb esta disponible.

    `imagen` es un tensor (3,H,W) o (N,3,H,W) en [0,1]. Con varias se suben
    todas, que es lo que hace falta para una interpolacion o un lote diverso.

    Aislado aqui por el mismo motivo que el resto del modulo: el codigo de
    interpretabilidad no debe importar wandb ni saber si esta instalado.
    """
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    lote = imagen if imagen.dim() == 4 else imagen.unsqueeze(0)
    run.log({clave: [wandb.Image(f, caption=titulo) for f in lote]}, step=paso)
