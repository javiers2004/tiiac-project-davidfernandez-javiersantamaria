"""Dos funciones. Es la capa que usa la aplicacion.

    neuronas = listar_neuronas("foto.jpg")        # foto -> que neuronas se activan
    imagen   = generar_imagen(neuronas[:3])       # neuronas -> foto
    sueno    = generar_deepdream("layer3")        # capa entera -> foto

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
from src.interpretability.objetivos import (
    Canal,
    Capa,
    Logit,
    Neurona,
    Probabilidad,
    Suma,
    resolver_modulo,
)

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "configs" / "feature_viz.yaml"

# Los objetivos que se construyen a partir de una Activacion (capa + canal).
# Layer no esta porque no elige unidad: va por generar_deepdream().
POR_NEURONA = {"canal": Canal, "neurona": Neurona, "logit": Logit,
               "probabilidad": Probabilidad}


METADATOS = RAIZ / "data" / "metadata"


@dataclass(frozen=True)
class Activacion:
    """Una neurona y cuanto la ha activado la foto."""

    capa: str
    canal: int
    valor: float

    def __str__(self) -> str:
        return f"{self.capa}[{self.canal}] = {self.valor:.3f}"


@dataclass(frozen=True)
class Prediccion:
    """Una clase predicha, con su nombre de especie si esta disponible."""

    clase: int
    probabilidad: float
    especie: str | None = None   # id de Pl@ntNet
    nombre: str | None = None    # nombre cientifico

    def __str__(self) -> str:
        quien = self.nombre or f"clase {self.clase}"
        return f"{quien} ({self.probabilidad:.1%})"


def cargar_especies() -> dict[int, tuple[str, str]]:
    """Mapa clase -> (id de especie, nombre cientifico).

    Devuelve {} si los metadatos no estan descargados, para que la aplicacion
    siga funcionando enseñando numeros de clase en vez de romperse. Los
    ficheros viven en data/, que no va al repositorio: se obtienen con
    `python scripts/get_data.py --solo-metadatos`.
    """
    idx = METADATOS / "class_idx_to_species_id.json"
    nombres = METADATOS / "plantnet300K_species_id_2_name.json"
    if not (idx.is_file() and nombres.is_file()):
        return {}

    import json
    mapa = json.loads(idx.read_text(encoding="utf-8"))
    nom = json.loads(nombres.read_text(encoding="utf-8"))
    return {int(k): (v, nom.get(v, "?")) for k, v in mapa.items()}


def cargar_modelo(cfg: dict | None = None) -> nn.Module:
    """Carga el checkpoint una vez, para no releerlo en cada llamada."""
    return construir_modelo_congelado((cfg or cargar_config(CONFIG))["modelo"])


def predecir(imagen, top_k: int = 1, modelo: nn.Module | None = None,
             cfg: dict | None = None) -> list[Prediccion]:
    """Las top_k especies mas probables para una foto.

    Si los metadatos no estan descargados, las predicciones llegan sin
    nombre: la clase sigue siendo correcta, solo que sin traducir.
    """
    cfg = cfg or cargar_config(CONFIG)
    modelo = modelo or cargar_modelo(cfg)
    with torch.no_grad():
        probas = modelo(_preparar(imagen, cfg)).softmax(dim=-1)

    especies = cargar_especies()
    valores, clases = probas[0].topk(min(top_k, probas.shape[-1]))
    salida = []
    for p, c in zip(valores, clases):
        sid, nombre = especies.get(int(c), (None, None))
        salida.append(Prediccion(int(c), float(p), sid, nombre))
    return salida


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
                   tipo: str = "canal",
                   pasos: int | None = None,
                   modelo: nn.Module | None = None,
                   cfg: dict | None = None) -> torch.Tensor:
    """La foto que mas activa a esas neuronas, partiendo de ruido.

    Args:
        neuronas: una Activacion o varias. Si son varias se maximizan a la
            vez, que es el objetivo "jointly optimized" del articulo.
        tipo: 'canal' promedia el canal entero (Channel del articulo);
            'neurona' usa solo la posicion central (Neuron). En las capas
            profundas apenas se distinguen, porque una sola posicion ya ve
            toda la imagen; en las tempranas la diferencia es enorme.
            'logit' maximiza el logit de la clase (Class Logits); las
            neuronas tienen que ser de 'fc', p.ej. las que devuelve
            listar_neuronas(foto, capas='fc'), cuyo canal es la clase.
            'probabilidad' maximiza la softmax de la clase (Class
            Probability), con las mismas neuronas que 'logit'. Sirve para
            comparar: tiende a hundir las demas clases en vez de dibujar
            la propia.
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

    if tipo not in POR_NEURONA:
        raise ValueError(f"tipo desconocido: '{tipo}'. "
                         f"Opciones: {', '.join(POR_NEURONA)}")

    cfg["objetivo"] = {"tipo": tipo, "capa": neuronas[0].capa,
                       "canal": neuronas[0].canal}
    if pasos is not None:
        cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(pasos)}

    construir = POR_NEURONA[tipo]
    objetivo = Suma(tuple(construir(n.canal) for n in neuronas))
    imagen, _ = maximizar_activacion(cfg, modelo=modelo, objetivo=objetivo)
    return imagen


def generar_deepdream(capa: str, pasos: int | None = None,
                      modelo: nn.Module | None = None,
                      cfg: dict | None = None) -> torch.Tensor:
    """La foto que mas excita una capa ENTERA, partiendo de ruido.

    Es el objetivo Layer (DeepDream) del articulo. No se eligen neuronas, por
    eso no pasa por generar_imagen(): se maximiza el cuadrado de todas las
    unidades de la capa y sale lo que esa profundidad de la red tiende a ver.

    Args:
        capa: el modulo, p.ej. 'layer3' o 'layer3.1.conv2'.
        pasos: pasos de optimizacion; por defecto los del YAML.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Tensor (1, 3, H, W) en [0, 1], listo para save_image().
    """
    cfg = cargar_config(CONFIG) if cfg is None else dict(cfg)
    cfg["objetivo"] = {"tipo": "capa", "capa": capa}
    if pasos is not None:
        cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(pasos)}

    imagen, _ = maximizar_activacion(cfg, modelo=modelo, objetivo=Capa())
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
