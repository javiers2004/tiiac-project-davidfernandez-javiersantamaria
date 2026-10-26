"""Dos funciones. Es la capa que usa la aplicacion.

    neuronas = listar_neuronas("foto.jpg")        # foto -> que neuronas se activan
    imagen   = generar_imagen(neuronas[:3])       # neuronas -> foto
    sueno    = generar_deepdream("layer3")        # capa entera -> foto
    otra     = generar_direccion("layer4", 3)     # direccion aleatoria -> foto
    fila     = generar_interpolacion(neuronas[0], neuronas[1], imagenes=6)
    facetas  = generar_diversas(neuronas[0], imagenes=4)  # varias facetas
    reales   = ejemplos_del_dataset(neuronas[0], "data/muestras")  # y fotos reales

Todo lo demas (parametrizacion de Fourier, robustez, regularizacion) vive
debajo con los valores de configs/feature_viz.yaml. Si no quieres tocarlo, no
lo tocas: estas dos funciones se apanan solas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Direccion,
    Interpolacion,
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
        por_canal = _por_canal(x, capa, modelo)
        valores, indices = por_canal.topk(min(top_k, por_canal.numel()))
        salida += [Activacion(capa, int(i), float(v))
                   for v, i in zip(valores, indices)]

    return sorted(salida, key=lambda a: a.valor, reverse=True)


def mapa_activacion(imagen, neurona: Activacion,
                    modelo: nn.Module | None = None,
                    cfg: dict | None = None,
                    recortar_negativos: bool = True) -> torch.Tensor:
    """Donde se enciende esa neurona DENTRO de la foto.

    Un canal de una capa convolucional no da un numero: da una rejilla (14x14
    en layer3 a 224 px) con un valor por cada trozo de la imagen.
    `listar_neuronas` promedia esa rejilla para poder ordenar canales; esta
    funcion devuelve la rejilla ANTES del promedio, estirada al tamano de la
    foto.

    Es el eslabon que falta entre el ranking y el patron optimizado: sin el,
    la interfaz salta de un numero a un dibujo que no se parece a la foto y
    nada explica que tienen que ver.

    Args:
        imagen: ruta a un fichero, o un tensor (1,3,H,W) en [0,1].
        neurona: la Activacion que devuelve listar_neuronas; lleva capa y canal.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.
        recortar_negativos: las capas pre-ReLU (layer3.1.conv2 y companeras)
            dan valores negativos, que significan que esa zona INHIBE el canal.
            Por defecto se recortan a cero, que es lo habitual en un mapa de
            calor. En False se conserva el signo y el mapa se normaliza
            alrededor de cero.

    Returns:
        Tensor (H, W) en [0, 1], del tamano de imagen.tamano, listo para
        pintarse como mapa de calor sobre la foto.
    """
    cfg = cfg or cargar_config(CONFIG)
    modelo = modelo or cargar_modelo(cfg)

    act = _activacion_de(_preparar(imagen, cfg), neurona.capa, modelo)
    if act.dim() != 4:
        raise ValueError(
            f"la capa '{neurona.capa}' no tiene extension espacial "
            f"({tuple(act.shape)}), asi que no hay nada que mapear. "
            f"Los mapas solo tienen sentido en capas convolucionales."
        )
    if not 0 <= neurona.canal < act.shape[1]:
        raise IndexError(f"la capa '{neurona.capa}' tiene {act.shape[1]} "
                         f"canales; el canal {neurona.canal} no existe")

    mapa = act[0, neurona.canal]
    # Un canal apagado del todo da un mapa constante. Sin el guardia del tope
    # saldria una division por cero: un mapa de NaN se pinta en negro y
    # parece un resultado valido.
    if recortar_negativos:
        mapa = mapa.clamp(min=0)
        tope = float(mapa.max())
        mapa = mapa / tope if tope > 0 else torch.zeros_like(mapa)
    else:
        # Escala simetrica: 0.5 es ausencia de respuesta, y los extremos
        # inhiben y excitan por igual.
        tope = float(mapa.abs().max())
        mapa = mapa / (2 * tope) + 0.5 if tope > 0 else torch.full_like(mapa, 0.5)

    tam = int(cfg["imagen"]["tamano"])
    return F.interpolate(mapa[None, None], size=(tam, tam),
                         mode="bilinear", align_corners=False)[0, 0]


def superponer_mapa(imagen, mapa: torch.Tensor, cfg: dict | None = None,
                    mezcla: float = 0.55) -> torch.Tensor:
    """Pinta el mapa de calor encima de la foto.

    Args:
        imagen: la misma foto que se le paso a mapa_activacion().
        mapa: lo que devuelve mapa_activacion(), (H, W) en [0, 1].
        cfg: configuracion; por defecto configs/feature_viz.yaml.
        mezcla: cuanto pesa el mapa frente a la foto, de 0 a 1.

    Returns:
        Tensor (3, H, W) en [0, 1], listo para save_image().
    """
    from matplotlib import cm          # pesa al importar; solo hace falta aqui

    cfg = cfg or cargar_config(CONFIG)
    tam = int(cfg["imagen"]["tamano"])

    foto = _a_tensor(imagen, tam)[0]
    calor = torch.as_tensor(cm.inferno(mapa.cpu().numpy())[..., :3],
                            dtype=torch.float32).permute(2, 0, 1)
    return (foto * (1 - mezcla) + calor * mezcla).clamp(0, 1)


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


def generar_direccion(capa: str, semilla: int, pasos: int | None = None,
                      modelo: nn.Module | None = None,
                      cfg: dict | None = None) -> torch.Tensor:
    """La foto que mas empuja la capa en una direccion ALEATORIA.

    Es el objetivo Direction de "The Interaction between Neurons": en vez de
    un canal (un eje del espacio de activaciones) se maximiza una combinacion
    aleatoria de todos. Cada semilla es una direccion distinta y siempre la
    misma. Segun el articulo, a menudo son interpretables, pero menos veces
    que las neuronas sueltas.

    Args:
        capa: el modulo, p.ej. 'layer4.1.conv2'.
        semilla: identifica la direccion.
        pasos: pasos de optimizacion; por defecto los del YAML.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Tensor (1, 3, H, W) en [0, 1], listo para save_image().
    """
    cfg = cargar_config(CONFIG) if cfg is None else dict(cfg)
    cfg["objetivo"] = {"tipo": "direccion", "capa": capa, "direccion": int(semilla)}
    if pasos is not None:
        cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(pasos)}

    imagen, _ = maximizar_activacion(cfg, modelo=modelo,
                                     objetivo=Direccion(int(semilla)))
    return imagen


def generar_interpolacion(desde: Activacion, hasta: Activacion, imagenes: int,
                          pasos: int | None = None,
                          modelo: nn.Module | None = None,
                          cfg: dict | None = None) -> torch.Tensor:
    """Una fila de fotos que va de una neurona a otra.

    Es la interpolacion de "The Interaction between Neurons": la primera
    imagen maximiza `desde`, la ultima `hasta` y las de en medio una mezcla
    lineal. La alineacion y la parametrizacion compartida que la acompanan
    se configuran en el YAML, como el resto.

    Args:
        desde: la neurona de partida, p.ej. una de listar_neuronas().
        hasta: la de llegada. Tiene que ser de la misma capa.
        imagenes: cuantas imagenes tiene la fila, 2 o mas.
        pasos: pasos de optimizacion; por defecto los del YAML.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Tensor (imagenes, 3, H, W) en [0, 1]. save_image(fila, f, nrow=imagenes)
        lo guarda como una fila.
    """
    if desde.capa != hasta.capa:
        raise ValueError(f"las dos neuronas tienen que ser de la misma capa; "
                         f"han llegado de '{desde.capa}' y '{hasta.capa}'")
    cfg = cargar_config(CONFIG) if cfg is None else dict(cfg)
    cfg["objetivo"] = {"tipo": "interpolacion", "capa": desde.capa,
                       "canales": [desde.canal, hasta.canal]}
    cfg["imagen"] = {**cfg["imagen"], "lote": int(imagenes)}
    if pasos is not None:
        cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(pasos)}

    fila, _ = maximizar_activacion(
        cfg, modelo=modelo, objetivo=Interpolacion(desde.canal, hasta.canal))
    return fila


def generar_diversas(neurona: Activacion, imagenes: int, tipo: str = "canal",
                     capa_diversidad: str | None = None,
                     peso: float | None = None,
                     pasos: int | None = None,
                     modelo: nn.Module | None = None,
                     cfg: dict | None = None) -> torch.Tensor:
    """Varias fotos distintas que activan la MISMA neurona.

    Es la seccion "Diversity" del articulo: una sola optimizacion ensena una
    faceta de lo que la neurona detecta; con el termino de diversidad las
    imagenes del lote se empujan a ser distintas entre si, y cada una puede
    ensenar otra. Sirve tambien para especies: tipo='logit' con una neurona
    de 'fc' (su canal es la clase) y capa_diversidad='layer4'.

    Args:
        neurona: la Activacion, p.ej. una de listar_neuronas().
        imagenes: cuantas versiones, 2 o mas.
        tipo: como en generar_imagen(): 'canal', 'neurona', 'logit'...
        capa_diversidad: donde se comparan las imagenes; por defecto la de
            la neurona. Obligatoria si la neurona es de 'fc'.
        peso: el del termino; por defecto el peso_sugerido del YAML.
        pasos: pasos de optimizacion; por defecto los del YAML.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Tensor (imagenes, 3, H, W) en [0, 1].
    """
    if int(imagenes) < 2:
        raise ValueError("la diversidad compara imagenes entre si: hacen falta 2 o mas")
    if tipo not in POR_NEURONA:
        raise ValueError(f"tipo desconocido: '{tipo}'. "
                         f"Opciones: {', '.join(POR_NEURONA)}")
    cfg = cargar_config(CONFIG) if cfg is None else dict(cfg)
    gcfg = dict(cfg["regularizacion"])
    dcfg = dict(gcfg.get("diversidad") or {})
    dcfg["peso"] = float(dcfg["peso_sugerido"] if peso is None else peso)
    if capa_diversidad is not None:
        dcfg["capa"] = capa_diversidad
    cfg["regularizacion"] = {**gcfg, "diversidad": dcfg}
    cfg["objetivo"] = {"tipo": tipo, "capa": neurona.capa, "canal": neurona.canal}
    cfg["imagen"] = {**cfg["imagen"], "lote": int(imagenes)}
    if pasos is not None:
        cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(pasos)}

    fotos, _ = maximizar_activacion(
        cfg, modelo=modelo, objetivo=POR_NEURONA[tipo](neurona.canal))
    return fotos


@dataclass(frozen=True)
class Ejemplo:
    """Una foto real y cuanto activa una neurona."""

    foto: Path
    valor: float

    def __str__(self) -> str:
        return f"{self.foto.name} = {self.valor:.3f}"


def ejemplos_del_dataset(neurona: Activacion, fotos: str | Path | list,
                         top_k: int = 6, modelo: nn.Module | None = None,
                         cfg: dict | None = None) -> list[Ejemplo]:
    """Las fotos reales que mas activan una neurona, de mas a menos.

    Es la contraparte de las visualizaciones que usa el articulo ("dataset
    examples"): con la diversidad salen varias facetas de una neurona, y
    mirar las fotos que mas la activan es como se comprueba si esas facetas
    existen de verdad o son artefactos de la optimizacion.

    Mide lo mismo que listar_neuronas(): la media espacial del canal, o el
    logit si la neurona es de 'fc'.

    Args:
        neurona: la Activacion; de ella solo se usan capa y canal.
        fotos: una carpeta (se toman sus .jpg, .jpeg y .png) o una lista de
            rutas. `python scripts/get_data.py --fotos N` baja unas cuantas.
        top_k: cuantas devolver.
        modelo: el modelo ya cargado.
        cfg: configuracion; por defecto configs/feature_viz.yaml.

    Returns:
        Las top_k fotos ordenadas de mayor a menor activacion.
    """
    cfg = cfg or cargar_config(CONFIG)
    modelo = modelo or cargar_modelo(cfg)
    if isinstance(fotos, (str, Path)):
        carpeta = Path(fotos)
        rutas = sorted(f for f in carpeta.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    else:
        rutas = [Path(f) for f in fotos]
    if not rutas:
        raise ValueError(f"no hay fotos en {fotos}. Se bajan con "
                         f"`python scripts/get_data.py --fotos N`.")

    valores = [float(_por_canal(_preparar(r, cfg), neurona.capa, modelo)[neurona.canal])
               for r in rutas]
    orden = sorted(zip(rutas, valores), key=lambda rv: rv[1], reverse=True)
    return [Ejemplo(r, v) for r, v in orden[:top_k]]


def _por_canal(x: torch.Tensor, capa: str, modelo: nn.Module) -> torch.Tensor:
    """Activacion de cada canal de `capa` para una entrada ya preparada: (C,).

    La media espacial de cada canal es exactamente el objetivo Channel,
    medido en vez de maximizado. En 'fc' es el logit de cada clase.
    """
    act = _activacion_de(x, capa, modelo)
    return act[0].mean(dim=(1, 2)) if act.dim() == 4 else act[0]
def _a_tensor(imagen, tam: int) -> torch.Tensor:
    """Fichero o tensor -> tensor (1,3,tam,tam) en [0,1], SIN normalizar.

    Es la foto tal cual se ve. La version normalizada, que es la que come el
    modelo, sale de _preparar().
    """
    if isinstance(imagen, (str, Path)):
        imagen = TF.to_tensor(Image.open(imagen).convert("RGB")).unsqueeze(0)
    if imagen.dim() == 3:
        imagen = imagen.unsqueeze(0)
    if imagen.shape[-2:] != (tam, tam):
        imagen = TF.resize(imagen, [tam, tam], antialias=True)
    return imagen


def _preparar(imagen, cfg: dict) -> torch.Tensor:
    """Fichero o tensor -> tensor normalizado como espera el modelo.

    Usa el MISMO tamano y las MISMAS estadisticas que la optimizacion. Si
    divergen, el ranking que ve el usuario no corresponde a lo que luego se
    optimiza, y no da ningun error.
    """
    ncfg = cfg["normalizacion"]
    media = torch.tensor(ncfg["media"]).view(1, 3, 1, 1)
    desv = torch.tensor(ncfg["desviacion"]).view(1, 3, 1, 1)
    return (_a_tensor(imagen, int(cfg["imagen"]["tamano"])) - media) / desv


def _activacion_de(x: torch.Tensor, capa: str, modelo: nn.Module) -> torch.Tensor:
    """Pasa la entrada ya normalizada y devuelve la activacion de esa capa.

    El hook se quita siempre, incluso si el forward falla: un hook olvidado
    se dispara en todas las llamadas posteriores y captura activaciones que
    nadie pidio.
    """
    capturado: dict[str, torch.Tensor] = {}
    asa = resolver_modulo(modelo, capa).register_forward_hook(
        lambda _m, _e, salida: capturado.__setitem__("act", salida)
    )
    try:
        with torch.no_grad():
            modelo(x)
    finally:
        asa.remove()
    return capturado["act"]
