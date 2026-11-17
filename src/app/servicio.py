"""Lo que hay detras de la interfaz: todo lo que pide la pagina, sin nada de HTTP.

serve.py solo traduce rutas a llamadas de esta clase. Asi se puede probar sin
levantar un servidor, y las reglas (que capa es cada bloque, que objetivo sale
de lo que se ha pulsado, como se reparten las capturas) viven en un solo sitio.

Carga checkpoints y responde: no entrena ni importa nada de src/training/.
Para la red usa la fachada src/interpretability/api.py y el bucle de
feature_viz.py, los mismos que la linea de comandos.

Hay dos copias de cada modelo. La del analisis atiende peticiones cortas (el
diagrama, los mapas); la de optimizar pasa minutos en un hilo con sus hooks
puestos. Si compartieran instancia, un analisis lanzado durante una
optimizacion dispararia los hooks del bucle y le cambiaria las activaciones
sin dar ningun error.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import yaml
from matplotlib import colormaps
from PIL import Image

from src.interpretability.api import (
    CONFIG,
    METADATOS,
    Activacion,
    _a_tensor,
    _preparar,
    cargar_especies,
    mapa_activacion,
    superponer_mapa,
)
from src.interpretability.feature_viz import (
    UnidadMuerta,
    _variacion_total,
    cargar_config,
    construir_modelo_congelado,
    maximizar_activacion,
    nombre_salida,
)
from src.interpretability.objetivos import (Canal, Capa, Logit, Neurona, Suma,
                                            resolver_modulo)
from src.utils.tracking import abrir_run

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_APP = RAIZ / "configs" / "app.yaml"
_INFERNO = colormaps["inferno"]
_EXTENSIONES = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------- estructura

def capas_por_etapa(modelo: nn.Module) -> list[list[str]]:
    """La ultima convolucion de cada bloque, agrupada por etapa (layer1..layer4).

    Es conv2 en el BasicBlock (ResNet-18) y conv3 en el Bottleneck (50 y 152):
    la salida del bloque antes de sumarle el atajo, que es pre-ReLU y no
    arranca muerta como la salida del bloque entero.
    """
    etapas = []
    for e in range(1, 5):
        nombres = []
        for b, bloque in enumerate(getattr(modelo, f"layer{e}")):
            conv = "conv3" if hasattr(bloque, "conv3") else "conv2"
            nombres.append(f"layer{e}.{b}.{conv}")
        etapas.append(nombres)
    return etapas


def campos_receptivos(modelo: nn.Module) -> dict[str, Any]:
    """Campo receptivo teorico de la ultima conv de cada bloque.

    Recorre las convoluciones y el max-pool del camino principal (sin los
    downsample, que van en paralelo y ven menos) acumulando kernel y stride.

    Returns:
        {'rf': lado en pixeles por etapa y bloque, 'inicio': centro de la
        posicion 0, 'j': salto entre posiciones vecinas por etapa}.
    """
    objetivo = {c: (e, b) for e, etapa in enumerate(capas_por_etapa(modelo))
                for b, c in enumerate(etapa)}
    tabla: dict[str, Any] = {"rf": [[], [], [], []], "inicio": [[], [], [], []],
                             "j": [0, 0, 0, 0]}
    rf, salto, inicio = 1, 1, 0.5
    for nombre, mod in modelo.named_modules():
        if "downsample" in nombre or not isinstance(mod, (nn.Conv2d, nn.MaxPool2d)):
            continue
        k, s, p = (v if isinstance(v, int) else v[0]
                   for v in (mod.kernel_size, mod.stride, mod.padding))
        rf += (k - 1) * salto
        inicio += ((k - 1) / 2 - p) * salto
        salto *= s
        if nombre in objetivo:
            e, _ = objetivo[nombre]
            tabla["rf"][e].append(rf)
            tabla["inicio"][e].append(inicio)
            tabla["j"][e] = salto
    return tabla


def columnas_rejilla(canales: int) -> int:
    """Columnas de la rejilla de canales del panel: 16, 32 o 64 segun cuantos haya."""
    return 16 if canales <= 128 else 32 if canales <= 512 else 64


def columnas_atlas(canales: int) -> int:
    """Columnas del atlas de mapas: la potencia de 2 que lo deja casi cuadrado."""
    return 2 ** math.ceil(math.log2(math.sqrt(canales)))


def colores(valores: np.ndarray) -> tuple[list[str], float, float]:
    """Color inferno de cada valor sobre el rango de la capa, y ese rango."""
    lo, hi = float(valores.min()), float(valores.max())
    t = (valores - lo) / (hi - lo) if hi > lo else np.zeros_like(valores)
    rgb = (_INFERNO(t)[:, :3] * 255).round().astype(int)
    return ["#%02x%02x%02x" % tuple(c) for c in rgb], lo, hi


def canales_de_fondo(act: torch.Tensor, cuantos: int = 24) -> list[int]:
    """Canales vivos repartidos por todo el rango, para las hojas de detras de la pila.

    Sin azar a proposito: las semillas solo se fijan en src/utils/seed.py, y la
    pila tiene que salir igual cada vez que se abre la misma foto.
    """
    vivos = [int(c) for c in torch.nonzero(act.flatten(1).amax(1) > 0).flatten()]
    if len(vivos) <= cuantos:
        return vivos
    return [vivos[int(k * len(vivos) / cuantos)] for k in range(cuantos)]


# ---------------------------------------------------------------- peticiones

def construir_objetivo(o: dict) -> Any:
    """El objetivo de feature_viz que corresponde a lo pulsado en la interfaz.

    Args:
        o: {'tipo': 'canal'|'neurona'|'capa'|'multi', 'capa', 'canal',
            'canales', 'x', 'y'}. En 'fc' un canal es una clase y se optimiza
            su logit, que es lo que el selector llama Channel sobre fc.
    """
    tipo, capa = o["tipo"], str(o["capa"])
    fc = capa == "fc"
    if tipo == "canal":
        return Logit(int(o["canal"])) if fc else Canal(int(o["canal"]))
    if tipo == "neurona":
        if fc:
            raise ValueError("fc no tiene extension espacial: el objetivo Neuron no se aplica")
        return Neurona(int(o["canal"]), int(o["x"]), int(o["y"]))
    if tipo == "capa":
        return Capa()
    if tipo == "multi":
        canales = [int(c) for c in o.get("canales") or []]
        if not canales:
            raise ValueError("Multiple channels necesita al menos un canal marcado")
        return Suma(tuple(Logit(c) if fc else Canal(c) for c in canales))
    raise ValueError(f"objetivo desconocido: {tipo}")


def describir_objetivo(o: dict) -> dict:
    """El bloque 'objetivo' del YAML equivalente: nombra el run y el fichero."""
    capa, tipo = str(o["capa"]), o["tipo"]
    if tipo == "capa":
        return {"tipo": "capa", "capa": capa}
    if tipo == "multi":
        return {"tipo": "suma", "capa": capa, "canales": [int(c) for c in o["canales"]]}
    if tipo == "neurona":
        return {"tipo": "neurona", "capa": capa, "canal": int(o["canal"]),
                "x": int(o["x"]), "y": int(o["y"])}
    return {"tipo": "logit" if capa == "fc" else "canal", "capa": capa, "canal": int(o["canal"])}


def construir_cfg(base: dict, peticion: dict, arquitectura: str, checkpoint: str) -> dict:
    """Configuracion de feature_viz con los mandos de la interfaz aplicados encima.

    Los mandos que no son un numero del YAML se traducen aqui: la rotacion
    maxima a la lista de angulos, la amplitud de escala a la lista de escalas
    y el interruptor de Fourier a la parametrizacion.
    """
    p = peticion["parametros"]
    cfg = copy.deepcopy(base)
    cfg["modelo"] = {**cfg["modelo"], "arquitectura": arquitectura, "checkpoint": checkpoint}
    cfg["experimento"] = {**cfg["experimento"], "seed": int(p["semilla"])}
    cfg["optimizacion"] = {**cfg["optimizacion"], "pasos": int(p["pasos"]),
                           "learning_rate": float(p["learning_rate"]),
                           "dispositivo": "cuda" if peticion.get("dispositivo") == "gpu" else "cpu"}
    cfg["regularizacion"] = {**cfg["regularizacion"], "l1": float(p["l1"]),
                             "tv": float(p["tv"]), "blur": float(p["blur"])}
    jitter, rot, amp = int(p["jitter"]), int(p["rotacion"]), float(p["escala"])
    cfg["robustez"] = {
        **cfg["robustez"], "activa": bool(p["robustez"]), "jitter": jitter,
        "jitter_final": jitter // 2, "rotaciones": list(range(-rot, rot + 1)),
        "escalas": [1.0] if amp == 0 else [round(1 + f * amp, 4) for f in (-1, -0.5, 0, 0.5, 1)],
    }
    cfg["imagen"] = {**cfg["imagen"], "lote": 1, "decay_power": float(p["decay_power"]),
                     "parametrizacion": "fourier" if p["fourier"] else "pixel"}
    cfg["objetivo"] = describir_objetivo(peticion["objetivo"])
    return cfg


def pasos_de_captura(pasos: int, capturas: int) -> list[int]:
    """Los pasos que se guardan: el total repartido en partes iguales (250/500/750/1000)."""
    return [round(pasos * (k + 1) / capturas) for k in range(capturas)]


def _png(arr: np.ndarray, nivel: int = 6) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=nivel)
    return buf.getvalue()


def _a_uint8(img: torch.Tensor) -> np.ndarray:
    """Tensor (1|3, H, W) o (1, 3, H, W) en [0, 1] -> array (H, W, 3) uint8."""
    if img.dim() == 4:
        img = img[0]
    return (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _data_url(img: torch.Tensor, formato: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.fromarray(_a_uint8(img)).save(buf, format=formato, **({"quality": 85} if formato == "JPEG" else {}))
    tipo = "png" if formato == "PNG" else "jpeg"
    return f"data:image/{tipo};base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- estado

@dataclass
class Foto:
    """Una foto subida o elegida, ya al tamano que come el modelo."""

    id: str
    nombre: str
    bytes: int
    tensor: torch.Tensor          # (1, 3, tam, tam) en [0, 1], sin normalizar


@dataclass
class Trabajo:
    """Una optimizacion en marcha o terminada, tal como la consulta la pagina."""

    id: str
    pasos: int
    capturas_en: list[int]
    referencia: float | None
    inicio: float = field(default_factory=time.time)
    fin: float | None = None
    paso: int = 0
    estado: str = "corriendo"
    error: str | None = None
    metricas: dict[str, list] = field(default_factory=lambda: {"activacion": [], "aspereza": [], "cambio": []})
    capturas: list[str | None] = field(default_factory=list)
    vista: str | None = None
    vista_paso: int = 0


class Servicio:
    """Fotos, analisis, mapas y optimizaciones para la interfaz.

    Args:
        cfg_app: configs/app.yaml ya leido; por defecto se lee del disco.
        cfg_viz: configs/feature_viz.yaml ya leido; por defecto se lee del disco.
        cargar: funcion arquitectura -> modelo congelado. Los tests inyectan
            un resnet sin entrenar; por defecto se lee el checkpoint.
    """

    def __init__(self, cfg_app: dict | None = None, cfg_viz: dict | None = None,
                 cargar: Callable[[str], nn.Module] | None = None, max_fotos: int = 6):
        self.app = cfg_app or yaml.safe_load(CONFIG_APP.read_text(encoding="utf-8"))
        self.viz = cfg_viz or cargar_config(CONFIG)
        self._inyectado = cargar is not None
        self._cargar = cargar or self._cargar_checkpoint
        self._max_fotos = max_fotos
        self._lock = threading.Lock()
        self._modelos: dict[tuple[str, str], tuple[nn.Module, threading.Lock]] = {}
        self._fotos: OrderedDict[str, Foto] = OrderedDict()
        self._analisis: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._trabajos: dict[str, Trabajo] = {}
        self._especies: list | None = None

    # ------------------------------------------------------------ modelos
    def _checkpoint(self, arq: str) -> str:
        return self.app["modelos"][arq]["checkpoint"]

    def _cargar_checkpoint(self, arq: str) -> nn.Module:
        mcfg = {**self.viz["modelo"], "arquitectura": arq, "checkpoint": self._checkpoint(arq)}
        return construir_modelo_congelado(mcfg)

    def disponible(self, arq: str) -> bool:
        """Si hay checkpoint para esa arquitectura (con un cargador inyectado, siempre)."""
        return self._inyectado or (RAIZ / self._checkpoint(arq)).is_file()

    def modelo(self, arq: str, uso: str = "analisis") -> tuple[nn.Module, threading.Lock]:
        """El modelo de esa arquitectura para ese uso, cargado una sola vez."""
        if arq not in self.app["modelos"]:
            raise ValueError(f"modelo desconocido: {arq}")
        if not self.disponible(arq):
            raise FileNotFoundError(f"falta {self._checkpoint(arq)}: python scripts/get_data.py --pesos {arq}")
        with self._lock:
            if (arq, uso) not in self._modelos:
                modelo = self._cargar(arq)
                # El analisis va siempre en CPU: _preparar normaliza con tensores de CPU,
                # y la copia de optimizar ya la mueve maximizar_activacion a donde toque.
                if uso == "analisis":
                    modelo = modelo.cpu()
                self._modelos[(arq, uso)] = (modelo, threading.Lock())
            return self._modelos[(arq, uso)]

    # ------------------------------------------------------------ info
    def especies(self) -> list | None:
        """[binomio, autor] de cada clase de fc, o None sin los metadatos."""
        if self._especies is None:
            mapa = cargar_especies()
            if mapa:
                n = max(mapa) + 1
                lista: list = [["clase %d" % i, ""] for i in range(n)]
                for clase, (_sid, nombre) in mapa.items():
                    partes = str(nombre).split(" ")
                    lista[clase] = [" ".join(partes[:2]), " ".join(partes[2:])]
                self._especies = lista
        return self._especies

    def muestras(self) -> list[dict]:
        """Las fotos de data/muestras, con el nombre de su especie si se conoce."""
        carpeta = RAIZ / self.app["muestras"]
        if not carpeta.is_dir():
            return []
        nombres = {}
        ruta = METADATOS / "plantnet300K_species_id_2_name.json"
        if ruta.is_file():
            nombres = json.loads(ruta.read_text(encoding="utf-8"))
        salida = []
        for f in sorted(carpeta.iterdir()):
            if f.suffix.lower() in _EXTENSIONES:
                especie = nombres.get(f.name.split("_")[0])
                salida.append({"nombre": f.name, "etiqueta": f"{especie} · {f.stem[:12]}" if especie else f.stem})
        return salida

    def valores_iniciales(self) -> dict:
        """Los mandos de la interfaz con los valores de configs/feature_viz.yaml."""
        v, r = self.viz, self.viz["robustez"]
        return {
            "pasos": int(v["optimizacion"]["pasos"]),
            "learning_rate": float(v["optimizacion"]["learning_rate"]),
            "semilla": int(v["experimento"]["seed"]),
            "l1": float(v["regularizacion"].get("l1", 0.0)),
            "tv": float(v["regularizacion"]["tv"]),
            "blur": float(v["regularizacion"].get("blur", 0.0)),
            "robustez": bool(r.get("activa", True)),
            "jitter": int(r["jitter"]),
            "rotacion": int(max((abs(a) for a in r["rotaciones"]), default=0)),
            "escala": round(max((abs(e - 1) for e in r["escalas"]), default=0.0), 4),
            "fourier": v["imagen"].get("parametrizacion", "fourier") == "fourier",
            "decay_power": float(v["imagen"].get("decay_power", 1.0)),
        }

    def info(self) -> dict:
        """Lo que la pagina necesita al arrancar."""
        w = self.viz.get("wandb", {})
        return {
            "modelos": {a: {"etiqueta": m["etiqueta"], "disponible": self.disponible(a)}
                        for a, m in self.app["modelos"].items()},
            "modelo_inicial": self.app["modelo_inicial"],
            "gpu": torch.cuda.is_available(),
            "capturas": int(self.app["capturas"]),
            "rangos": self.app["rangos"],
            "valores": self.valores_iniciales(),
            "muestras": self.muestras(),
            "wandb": f"https://wandb.ai/{w.get('entidad', '')}/{w.get('proyecto', '')}",
            "especies": self.especies(),
        }

    # ------------------------------------------------------------ fotos
    def anadir_foto(self, datos: bytes, nombre: str) -> dict:
        """Registra una foto (los bytes de un JPG o PNG) y devuelve su id."""
        try:
            img = Image.open(io.BytesIO(datos)).convert("RGB")
        except Exception as e:  # noqa: BLE001 - cualquier fallo de PIL es "no es una imagen"
            raise ValueError(f"no se puede leer {nombre} como imagen: {e}") from e
        fid = hashlib.sha1(datos).hexdigest()[:16]
        tam = int(self.viz["imagen"]["tamano"])
        tensor = _a_tensor(TF.to_tensor(img).unsqueeze(0), tam)
        with self._lock:
            self._fotos[fid] = Foto(fid, nombre, len(datos), tensor)
            self._fotos.move_to_end(fid)
            while len(self._fotos) > self._max_fotos:
                self._fotos.popitem(last=False)
        return {"id": fid, "nombre": nombre, "bytes": len(datos)}

    def anadir_muestra(self, nombre: str) -> dict:
        """Registra una foto de data/muestras por su nombre de fichero."""
        if nombre not in {m["nombre"] for m in self.muestras()}:
            raise ValueError(f"no hay ninguna muestra llamada {nombre}")
        ruta = RAIZ / self.app["muestras"] / nombre
        return self.anadir_foto(ruta.read_bytes(), nombre)

    def anadir_inicial(self) -> dict:
        """Registra la foto con la que arranca la aplicacion."""
        ruta = RAIZ / self.app["foto_inicial"]
        return self.anadir_foto(ruta.read_bytes(), ruta.name)

    def foto(self, fid: str) -> Foto:
        """La foto registrada con ese id."""
        if fid not in self._fotos:
            raise KeyError(f"no hay ninguna foto {fid}; vuelve a subirla")
        return self._fotos[fid]

    def foto_png(self, fid: str) -> bytes:
        """La foto tal como la ve el modelo (cuadrada, al tamano de entrada)."""
        return _png(_a_uint8(self.foto(fid).tensor))

    def canales_png(self, fid: str) -> bytes:
        """Sus canales R, G y B uno al lado del otro, cada uno en su color."""
        t = self.foto(fid).tensor[0]
        t = torch.nn.functional.interpolate(t[None], size=(112, 112), mode="bilinear", align_corners=False)[0]
        hoja = torch.zeros(3, 112, 336)
        for k in range(3):
            hoja[k, :, k * 112:(k + 1) * 112] = t[k]
        return _png(_a_uint8(hoja))

    # ------------------------------------------------------------ analisis
    def _activaciones(self, fid: str, arq: str) -> tuple[dict[str, torch.Tensor], nn.Module]:
        """Activacion de todos los bloques y de fc en una sola pasada."""
        modelo, lock = self.modelo(arq)
        capas = [c for etapa in capas_por_etapa(modelo) for c in etapa] + ["fc"]
        x = _preparar(self.foto(fid).tensor, self.viz)
        capturado: dict[str, torch.Tensor] = {}
        with lock:
            asas = [resolver_modulo(modelo, c).register_forward_hook(
                lambda _m, _e, s, c=c: capturado.__setitem__(c, s.detach().cpu()[0]))
                for c in capas]
            try:
                with torch.no_grad():
                    modelo(x)
            finally:
                for a in asas:
                    a.remove()
        return capturado, modelo

    def analizar(self, fid: str, arq: str) -> dict:
        """Todo lo que dibuja el diagrama para esa foto y ese modelo.

        Por etapa y por bloque: activacion media de cada canal, su color, los
        mas activos, el canal por defecto y canales para el fondo de la pila.
        Ademas el top-5 de especies, la forma de cada etapa y su campo receptivo.
        """
        clave = (fid, arq)
        if clave in self._analisis:
            self._analisis.move_to_end(clave)
            return self._analisis[clave]["json"]
        acts, modelo = self._activaciones(fid, arq)
        etapas = []
        for nombres in capas_por_etapa(modelo):
            bloques = []
            for capa in nombres:
                a = acts[capa]
                valores = a.mean(dim=(1, 2)).numpy()
                orden = np.argsort(-valores)
                cols, lo, hi = colores(valores)
                bloques.append({"capa": capa, "valores": [round(float(v), 3) for v in valores],
                                "colores": cols, "top8": [int(i) for i in orden[:8]],
                                "elegido": int(orden[0]), "lo": round(lo, 3), "hi": round(hi, 3),
                                "fondo": canales_de_fondo(a)})
            c, h, w = acts[nombres[-1]].shape
            ac = columnas_atlas(c)
            etapas.append({**bloques[-1], "corta": nombres[-1].split(".")[0], "canales": c, "lado": h,
                           "forma": f"{c}×{h}×{w}", "cols": columnas_rejilla(c),
                           "atlas": {"cols": ac, "filas": c // ac}, "bloques": bloques})
        logits = acts["fc"].numpy()
        orden = np.argsort(-logits)
        cols, lo, hi = colores(logits)
        etapas.append({"capa": "fc", "corta": "fc", "canales": int(len(logits)), "lado": 0,
                       "forma": f"{len(logits)} clases", "cols": 47,
                       "valores": [round(float(v), 3) for v in logits], "colores": cols,
                       "top8": [int(i) for i in orden[:8]], "elegido": int(orden[0]),
                       "lo": round(lo, 3), "hi": round(hi, 3)})
        prob = torch.softmax(acts["fc"], 0)
        top = torch.topk(prob, 5)
        datos = {"arquitectura": arq, "etapas": etapas,
                 "top5": [[int(c), round(float(p), 4)] for p, c in zip(top.values, top.indices)],
                 "campos": campos_receptivos(modelo)}
        with self._lock:
            self._analisis[clave] = {"json": datos, "acts": acts, "atlas": {}}
            while len(self._analisis) > self._max_fotos:
                self._analisis.popitem(last=False)
        return datos

    def _entrada(self, fid: str, arq: str) -> dict:
        self.analizar(fid, arq)
        return self._analisis[(fid, arq)]

    def atlas_png(self, fid: str, arq: str, etapa: int) -> bytes:
        """Los mapas de todos los canales de todos los bloques de una etapa, en un PNG.

        Cada bloque es un mosaico de columnas_atlas(C) columnas, y los bloques
        van uno debajo de otro: la pagina recorta el canal y el bloque que toque
        con background-position. Cada mapa se normaliza a su propio maximo (los
        negativos a cero) con raiz cuadrada, porque son dispersos y en lineal
        casi todo sale negro.
        """
        entrada = self._entrada(fid, arq)
        if etapa in entrada["atlas"]:
            return entrada["atlas"][etapa]
        nombres = [b["capa"] for b in entrada["json"]["etapas"][etapa]["bloques"]]
        mosaicos = []
        for capa in nombres:
            a = entrada["acts"][capa].clamp(min=0)
            c, h, w = a.shape
            tope = a.amax(dim=(1, 2), keepdim=True)
            t = torch.where(tope > 0, (a / tope.clamp(min=1e-12)).sqrt(), torch.zeros_like(a)).numpy()
            rgb = (_INFERNO(t)[..., :3] * 255).astype(np.uint8)            # (C, H, W, 3)
            cols = columnas_atlas(c)
            rgb = rgb.reshape(c // cols, cols, h, w, 3).transpose(0, 2, 1, 3, 4).reshape(c // cols * h, cols * w, 3)
            mosaicos.append(rgb)
        png = _png(np.concatenate(mosaicos, axis=0), nivel=1)
        entrada["atlas"][etapa] = png
        return png

    def mapa(self, fid: str, arq: str, capa: str, canal: int) -> dict:
        """La foto con el mapa de ese canal superpuesto, y el maximo de la escala."""
        entrada = self._entrada(fid, arq)
        if capa not in entrada["acts"] or capa == "fc":
            raise ValueError(f"{capa} no es una capa con mapa espacial de este modelo")
        a = entrada["acts"][capa]
        if not 0 <= canal < a.shape[0]:
            raise ValueError(f"{capa} tiene {a.shape[0]} canales; el {canal} no existe")
        modelo, lock = self.modelo(arq)
        foto = self.foto(fid).tensor
        with lock:
            mapa = mapa_activacion(foto, Activacion(capa, canal, 0.0), modelo, self.viz)
        imagen = superponer_mapa(foto, mapa, self.viz)
        return {"imagen": _data_url(imagen), "max": round(float(a[canal].clamp(min=0).max()), 3)}

    # ------------------------------------------------------------ optimizacion
    def lanzar(self, peticion: dict) -> str:
        """Arranca una optimizacion en segundo plano y devuelve su id.

        Args:
            peticion: {'foto', 'modelo', 'dispositivo': 'cpu'|'gpu', 'wandb',
                'objetivo': ver construir_objetivo, 'parametros': los mandos}.
        """
        if any(t.estado == "corriendo" for t in self._trabajos.values()):
            raise ValueError("ya hay una optimizacion en marcha")
        arq = peticion["modelo"]
        if peticion.get("dispositivo") == "gpu" and not torch.cuda.is_available():
            raise ValueError("no hay GPU disponible")
        objetivo = construir_objetivo(peticion["objetivo"])
        cfg = construir_cfg(self.viz, peticion, arq, self._checkpoint(arq))
        # Lo que activa la foto real ese mismo objetivo: la linea gris de la grafica.
        acts = self._entrada(peticion["foto"], arq)["acts"]
        capa = str(peticion["objetivo"]["capa"])
        referencia = float(objetivo(acts[capa][None])) if capa in acts else None
        pasos = int(cfg["optimizacion"]["pasos"])
        trabajo = Trabajo(uuid.uuid4().hex[:12], pasos,
                          pasos_de_captura(pasos, int(self.app["capturas"])), referencia)
        trabajo.capturas = [None] * len(trabajo.capturas_en)
        self._trabajos[trabajo.id] = trabajo
        hilo = threading.Thread(target=self._correr, args=(trabajo, cfg, objetivo, arq,
                                                           bool(peticion.get("wandb"))), daemon=True)
        hilo.start()
        return trabajo.id

    def _correr(self, t: Trabajo, cfg: dict, objetivo: Any, arq: str, wandb: bool) -> None:
        cada_vista = max(1, t.pasos // 64)
        previa: list[torch.Tensor | None] = [None]

        def al_progresar(paso: int, total: int, valor: float, vista: torch.Tensor) -> bool:
            t.metricas["activacion"].append(round(float(valor), 5))
            t.metricas["aspereza"].append(round(float(_variacion_total(vista)), 6))
            t.metricas["cambio"].append(None if previa[0] is None
                                        else round(float((vista - previa[0]).abs().mean()), 7))
            previa[0] = vista
            hecho = paso + 1
            if hecho in t.capturas_en:
                t.capturas[t.capturas_en.index(hecho)] = _data_url(vista)
            if paso % cada_vista == 0 or hecho == total:
                t.vista, t.vista_paso = _data_url(vista, "JPEG"), hecho
            t.paso = hecho
            return True

        run = None
        try:
            modelo, _ = self.modelo(arq, uso="optimizar")
            run = abrir_run(cfg, nombre_salida(cfg["objetivo"])) if wandb else None
            maximizar_activacion(cfg, run, modelo=modelo, objetivo=objetivo, al_progresar=al_progresar)
            t.estado = "hecho"
        except UnidadMuerta as e:
            t.estado, t.error = "error", str(e)
        except Exception as e:  # noqa: BLE001 - el error se ensena en la pagina en vez de perderse en el hilo
            t.estado, t.error = "error", f"{type(e).__name__}: {e}"
        finally:
            if run is not None:
                run.finish()
            t.fin = time.time()

    def estado(self, tid: str, desde: int = 0, capturas: int = 0, vista: int = 0) -> dict:
        """Lo nuevo de una optimizacion desde lo que la pagina ya tiene.

        Args:
            desde: cuantos pasos de metricas tiene ya.
            capturas: cuantas capturas tiene ya (llegan en orden).
            vista: el paso de la ultima vista previa que tiene.
        """
        if tid not in self._trabajos:
            raise KeyError(f"no hay ninguna optimizacion {tid}")
        t = self._trabajos[tid]
        # Solo los pasos que ya tienen las tres metricas: el hilo las anade una detras de otra.
        n = min(len(v) for v in t.metricas.values())
        listas = [c for c in t.capturas if c is not None]
        return {
            "estado": t.estado, "error": t.error, "paso": t.paso, "pasos": t.pasos,
            "segundos": round((t.fin or time.time()) - t.inicio, 2),
            "referencia": t.referencia, "capturas_en": t.capturas_en,
            "desde": desde, "metricas": {k: v[desde:n] for k, v in t.metricas.items()},
            "capturas": listas[capturas:],
            "vista": t.vista if t.vista_paso > vista else None, "vista_paso": t.vista_paso,
        }
