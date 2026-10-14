"""Dos funciones. Es la capa que usa la aplicacion.

    neuronas = listar_neuronas("foto.jpg")        # foto -> que neuronas se activan
    imagen   = generar_imagen(neuronas[:3])       # neuronas -> foto

Todo lo demas (parametrizacion de Fourier, robustez, regularizacion) vive
debajo con los valores de configs/feature_viz.yaml. Si no quieres tocarlo, no
lo tocas: estas dos funciones se apanan solas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from src.interpretability.feature_viz import (
    cargar_config,
    construir_modelo_congelado,
    maximizar_activacion,
)
from src.interpretability.objetivos import Canal, Suma, resolver_modulo

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "configs" / "feature_viz.yaml"


@dataclass(frozen=True)
class Activacion:
    """Una neurona y cuanto la ha activado la foto."""

    capa: str
    canal: int
    valor: float

    def __str__(self) -> str:
        return f"{self.capa}[{self.canal}] = {self.valor:.3f}"


def cargar_modelo(cfg: dict | None = None) -> nn.Module:
    """Carga el checkpoint una vez, para no releerlo en cada llamada."""
    return construir_modelo_congelado((cfg or cargar_config(CONFIG))["modelo"])


def predecir(imagen, modelo: nn.Module | None = None,
             cfg: dict | None = None) -> tuple[int, float]:
    """Devuelve (clase predicha, probabilidad) para una foto."""
    cfg = cfg or cargar_config(CONFIG)
    modelo = modelo or cargar_modelo(cfg)
    with torch.no_grad():
        probas = modelo(_preparar(imagen, cfg)).softmax(dim=-1)
    prob, clase = probas[0].max(dim=0)
    return int(clase), float(prob)


def listar_neuronas(imagen, capas: str | list[str] = "layer4", top_k: int = 10,
                    modelo: nn.Module | None = None,
                    cfg: dict | None = None) -> list[Activacion]:
    """Que neuronas activa esta foto, de mas a menos.

    Args:
        imagen: ruta a un fichero, o un tensor (1,3,H,W) en [0,1].
        capas: una capa o varias ('layer4', o ['layer3', 'layer4']).
        top_k: cuantas devolver POR CAPA.
        modelo: el modelo ya cargado, para no releer el checkpoint.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Las activaciones ordenadas de mayor a menor. Cada una lleva capa,
        canal y valor, y se le puede pasar tal cual a generar_imagen().

    Aviso: el orden solo es comparable DENTRO de una capa. Capas distintas
    tienen escalas distintas (layer1 activa de media 4 veces mas que layer3),
    asi que mezclarlas en un ranking unico compara peras con manzanas.
    """
    cfg = cfg or cargar_config(CONFIG)
    modelo = modelo or cargar_modelo(cfg)
    if isinstance(capas, str):
        capas = [capas]

    x = _preparar(imagen, cfg)
    salida: list[Activacion] = []

    for capa in capas:
        capturado: dict[str, torch.Tensor] = {}
        asa = resolver_modulo(modelo, capa).register_forward_hook(
            lambda _m, _e, s: capturado.__setitem__("act", s)
        )
        try:
            with torch.no_grad():
                modelo(x)
        finally:
            asa.remove()

        act = capturado["act"]
        # La media espacial de cada canal: es exactamente el objetivo Channel,
        # medido en vez de maximizado.
        por_canal = act[0].mean(dim=(1, 2)) if act.dim() == 4 else act[0]
        valores, indices = por_canal.topk(min(top_k, por_canal.numel()))
        salida += [Activacion(capa, int(i), float(v))
                   for v, i in zip(valores, indices)]

    return sorted(salida, key=lambda a: a.valor, reverse=True)


def generar_imagen(neuronas: Activacion | list[Activacion],
                   pasos: int | None = None,
                   modelo: nn.Module | None = None,
                   cfg: dict | None = None) -> torch.Tensor:
    """La foto que mas activa a esas neuronas, partiendo de ruido.

    Args:
        neuronas: una Activacion o varias. Si son varias se maximizan a la
            vez, que es el objetivo "jointly optimized" del articulo.
        pasos: pasos de optimizacion; por defecto los del YAML.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Tensor (1, 3, H, W) en [0, 1], listo para save_image().
    """
    cfg = cargar_config(CONFIG) if cfg is None else dict(cfg)
    if isinstance(neuronas, Activacion):
        neuronas = [neuronas]
    if not neuronas:
        raise ValueError("hay que pasar al menos una neurona")

    capas = {n.capa for n in neuronas}
    if len(capas) > 1:
        raise ValueError(
            f"de momento todas las neuronas tienen que ser de la misma capa; "
            f"han llegado de {sorted(capas)}. El bucle engancha una sola capa."
        )

    cfg["objetivo"] = {"tipo": "canal", "capa": neuronas[0].capa,
                       "canal": neuronas[0].canal}
    if pasos is not None:
        cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(pasos)}

    objetivo = Suma(tuple(Canal(n.canal) for n in neuronas))
    imagen, _ = maximizar_activacion(cfg, modelo=modelo, objetivo=objetivo)
    return imagen


def _preparar(imagen, cfg: dict) -> torch.Tensor:
    """Fichero o tensor -> tensor normalizado como espera el modelo.

    Usa el MISMO tamano y las MISMAS estadisticas que la optimizacion. Si
    divergen, el ranking que ve el usuario no corresponde a lo que luego se
    optimiza, y no da ningun error.
    """
    tam = int(cfg["imagen"]["tamano"])
    if isinstance(imagen, (str, Path)):
        imagen = TF.to_tensor(Image.open(imagen).convert("RGB")).unsqueeze(0)
    if imagen.dim() == 3:
        imagen = imagen.unsqueeze(0)
    if imagen.shape[-2:] != (tam, tam):
        imagen = TF.resize(imagen, [tam, tam], antialias=True)

    ncfg = cfg["normalizacion"]
    media = torch.tensor(ncfg["media"]).view(1, 3, 1, 1)
    desv = torch.tensor(ncfg["desviacion"]).view(1, 3, 1, 1)
    return (imagen - media) / desv
