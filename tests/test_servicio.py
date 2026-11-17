"""Tests del aplicativo: el servicio (src/app/servicio.py) y sus rutas HTTP.

No usan los .tar: un resnet sin entrenar basta, porque lo que se prueba es la
mecanica (que capa es cada bloque, que objetivo sale de cada peticion, que
forma tienen el analisis, los atlas y el progreso de una optimizacion), no
que el modelo reconozca plantas.
"""

import io
import json
import threading
import time
import urllib.request

import pytest
import torch
from PIL import Image
from torchvision.models import resnet18, resnet50

from src.app.serve import crear_servidor
from src.app.servicio import (
    Servicio,
    campos_receptivos,
    capas_por_etapa,
    columnas_atlas,
    columnas_rejilla,
    construir_cfg,
    construir_objetivo,
    pasos_de_captura,
)
from src.interpretability.api import CONFIG
from src.interpretability.feature_viz import cargar_config
from src.interpretability.objetivos import Canal, Capa, Logit, Neurona, Suma
from src.utils.seed import fijar_seed


def _congelado(m: torch.nn.Module) -> torch.nn.Module:
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def servicio():
    """Un servicio con un resnet18 sin entrenar en vez del checkpoint."""
    fijar_seed(0)
    estado = resnet18(num_classes=1081).state_dict()

    def cargar(_arq):
        m = resnet18(num_classes=1081)
        m.load_state_dict(estado)
        return _congelado(m)

    return Servicio(cargar=cargar)


@pytest.fixture(scope="module")
def foto(servicio):
    img = Image.new("RGB", (300, 260))
    for x in range(300):
        for y in range(260):
            img.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return servicio.anadir_foto(buf.getvalue(), "prueba.png")


def _parametros(servicio, **cambios):
    p = servicio.valores_iniciales()
    p.update(cambios)
    return p


# ------------------------------------------------------------ estructura

def test_capas_por_etapa_resnet18_y_resnet50():
    assert capas_por_etapa(resnet18()) == [[f"layer{e}.{b}.conv2" for b in range(2)] for e in range(1, 5)]
    c50 = capas_por_etapa(resnet50())
    assert [len(e) for e in c50] == [3, 4, 6, 3]
    assert c50[2][-1] == "layer3.5.conv3"


def test_campos_receptivos_conocidos():
    """Los valores publicados para ResNet-18 y ResNet-50 (ultimo bloque de cada etapa)."""
    c18 = campos_receptivos(resnet18())
    assert [rf[-1] for rf in c18["rf"]] == [43, 99, 211, 435]
    assert c18["j"] == [4, 8, 16, 32]
    c50 = campos_receptivos(resnet50())
    assert [rf[-1] for rf in c50["rf"]] == [35, 91, 267, 427]
    assert [len(rf) for rf in c50["rf"]] == [3, 4, 6, 3]


def test_columnas():
    assert [columnas_rejilla(c) for c in (64, 128, 256, 512, 1024, 2048)] == [16, 16, 32, 32, 64, 64]
    assert [columnas_atlas(c) for c in (64, 128, 256, 512, 1024, 2048)] == [8, 16, 16, 32, 32, 64]


def test_pasos_de_captura_se_reparten():
    assert pasos_de_captura(1000, 4) == [250, 500, 750, 1000]
    assert pasos_de_captura(256, 4) == [64, 128, 192, 256]


# ------------------------------------------------------------ peticiones

@pytest.mark.parametrize("o, esperado", [
    ({"tipo": "canal", "capa": "layer3.1.conv2", "canal": 7}, Canal(7)),
    ({"tipo": "canal", "capa": "fc", "canal": 248}, Logit(248)),
    ({"tipo": "neurona", "capa": "layer2.1.conv2", "canal": 3, "x": 1, "y": 2}, Neurona(3, 1, 2)),
    ({"tipo": "capa", "capa": "layer4.1.conv2"}, Capa()),
    ({"tipo": "multi", "capa": "layer4.1.conv2", "canales": [1, 2]}, Suma((Canal(1), Canal(2)))),
    ({"tipo": "multi", "capa": "fc", "canales": [5]}, Suma((Logit(5),))),
])
def test_construir_objetivo(o, esperado):
    assert construir_objetivo(o) == esperado


@pytest.mark.parametrize("o", [
    {"tipo": "neurona", "capa": "fc", "canal": 1, "x": 0, "y": 0},
    {"tipo": "multi", "capa": "layer4.1.conv2", "canales": []},
    {"tipo": "otro", "capa": "layer4.1.conv2"},
])
def test_construir_objetivo_rechaza(o):
    with pytest.raises(ValueError):
        construir_objetivo(o)


def test_construir_cfg_traduce_los_mandos():
    base = cargar_config(CONFIG)
    peticion = {
        "dispositivo": "cpu",
        "objetivo": {"tipo": "multi", "capa": "layer4.1.conv2", "canales": [3, 9]},
        "parametros": {"pasos": 64, "learning_rate": 0.01, "semilla": 7, "l1": 0.5, "tv": 2.0,
                       "blur": 10.0, "robustez": True, "jitter": 8, "rotacion": 2, "escala": 0.1,
                       "fourier": False, "decay_power": 1.5},
    }
    antes = json.dumps(base, sort_keys=True)
    cfg = construir_cfg(base, peticion, "resnet50", "models/x.tar")
    assert cfg["modelo"]["arquitectura"] == "resnet50"
    assert cfg["optimizacion"]["pasos"] == 64 and cfg["optimizacion"]["dispositivo"] == "cpu"
    assert cfg["experimento"]["seed"] == 7
    assert cfg["robustez"]["rotaciones"] == [-2, -1, 0, 1, 2]
    assert cfg["robustez"]["escalas"] == [0.9, 0.95, 1.0, 1.05, 1.1]
    assert cfg["robustez"]["jitter_final"] == 4
    assert cfg["imagen"]["parametrizacion"] == "pixel" and cfg["imagen"]["decay_power"] == 1.5
    assert cfg["objetivo"] == {"tipo": "suma", "capa": "layer4.1.conv2", "canales": [3, 9]}
    # la base no se toca: cada peticion trabaja sobre una copia
    assert json.dumps(base, sort_keys=True) == antes


def test_valores_iniciales_salen_del_yaml(servicio):
    v = servicio.valores_iniciales()
    viz = cargar_config(CONFIG)
    assert v["pasos"] == viz["optimizacion"]["pasos"]
    assert v["semilla"] == viz["experimento"]["seed"]
    assert v["rotacion"] == max(abs(a) for a in viz["robustez"]["rotaciones"])


# ------------------------------------------------------------ analisis

def test_analizar_tiene_la_forma_del_diagrama(servicio, foto):
    d = servicio.analizar(foto["id"], "resnet18")
    assert len(d["etapas"]) == 5
    assert [e["canales"] for e in d["etapas"]] == [64, 128, 256, 512, 1081]
    assert [e["lado"] for e in d["etapas"]] == [56, 28, 14, 7, 0]
    for e in d["etapas"][:4]:
        assert len(e["bloques"]) == 2
        assert e["capa"] == e["bloques"][-1]["capa"]
        assert len(e["colores"]) == len(e["valores"]) == e["canales"]
        assert e["atlas"]["cols"] * e["atlas"]["filas"] == e["canales"]
    assert len(d["top5"]) == 5
    assert abs(sum(p for _, p in d["top5"])) <= 1.0001


def test_atlas_apila_los_bloques(servicio, foto):
    d = servicio.analizar(foto["id"], "resnet18")
    png = servicio.atlas_png(foto["id"], "resnet18", 0)
    img = Image.open(io.BytesIO(png))
    e = d["etapas"][0]
    assert img.size == (e["atlas"]["cols"] * 56, e["atlas"]["filas"] * 56 * len(e["bloques"]))


def test_mapa_superpuesto(servicio, foto):
    m = servicio.mapa(foto["id"], "resnet18", "layer3.1.conv2", 5)
    assert m["imagen"].startswith("data:image/png;base64,")
    assert m["max"] >= 0
    with pytest.raises(ValueError):
        servicio.mapa(foto["id"], "resnet18", "fc", 5)


def test_foto_desconocida(servicio):
    with pytest.raises(KeyError):
        servicio.analizar("0" * 16, "resnet18")


# ------------------------------------------------------------ optimizacion

def _esperar(servicio, tid, maximo=120):
    t0 = time.time()
    while time.time() - t0 < maximo:
        e = servicio.estado(tid)
        if e["estado"] != "corriendo":
            return e
        time.sleep(0.1)
    raise TimeoutError(tid)


@pytest.mark.parametrize("objetivo", [
    {"tipo": "canal", "capa": "layer3.1.conv2", "canal": 4},
    {"tipo": "neurona", "capa": "layer2.1.conv2", "canal": 2, "x": 3, "y": 20},
    {"tipo": "capa", "capa": "layer4.1.conv2"},
    {"tipo": "multi", "capa": "fc", "canales": [1, 2]},
])
def test_optimizacion_completa(servicio, foto, objetivo):
    tid = servicio.lanzar({"foto": foto["id"], "modelo": "resnet18", "dispositivo": "cpu",
                           "objetivo": objetivo, "parametros": _parametros(servicio, pasos=8)})
    e = _esperar(servicio, tid)
    assert e["estado"] == "hecho", e["error"]
    assert e["paso"] == 8
    assert all(len(v) == 8 for v in e["metricas"].values())
    assert e["metricas"]["cambio"][0] is None and e["metricas"]["cambio"][1] >= 0
    assert e["capturas_en"] == [2, 4, 6, 8] and len(e["capturas"]) == 4
    assert e["referencia"] is not None


def test_estado_solo_devuelve_lo_nuevo(servicio, foto):
    tid = servicio.lanzar({"foto": foto["id"], "modelo": "resnet18", "dispositivo": "cpu",
                           "objetivo": {"tipo": "canal", "capa": "layer4.1.conv2", "canal": 0},
                           "parametros": _parametros(servicio, pasos=8)})
    _esperar(servicio, tid)
    e = servicio.estado(tid, desde=5, capturas=3, vista=8)
    assert all(len(v) == 3 for v in e["metricas"].values())
    assert len(e["capturas"]) == 1 and e["vista"] is None


# ------------------------------------------------------------ HTTP

@pytest.fixture(scope="module")
def url(servicio):
    servidor = crear_servidor(servicio, "127.0.0.1", 0)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{servidor.server_port}"
    servidor.shutdown()
    servidor.server_close()


def _pedir(url, metodo="GET", cuerpo=None, cabeceras=None):
    req = urllib.request.Request(url, data=cuerpo, method=metodo, headers=cabeceras or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_rutas(url, foto):
    estado, tipo, cuerpo = _pedir(url + "/")
    assert estado == 200 and "text/html" in tipo and b'<div id="app">' in cuerpo
    estado, tipo, _ = _pedir(url + "/static/app.js")
    assert estado == 200 and "javascript" in tipo
    estado, _, cuerpo = _pedir(url + "/api/info")
    info = json.loads(cuerpo)
    assert estado == 200 and set(info["modelos"]) == {"resnet18", "resnet50", "resnet152"}
    estado, tipo, _ = _pedir(f"{url}/api/fotos/{foto['id']}/foto.png")
    assert estado == 200 and tipo == "image/png"
    estado, _, cuerpo = _pedir(f"{url}/api/fotos/{foto['id']}/resnet18/analisis")
    assert estado == 200 and len(json.loads(cuerpo)["etapas"]) == 5


def test_subida_y_errores(url):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 30)).save(buf, format="JPEG")
    estado, _, cuerpo = _pedir(url + "/api/fotos", "POST", buf.getvalue(), {"X-Nombre": "hoja.jpg"})
    assert estado == 200 and json.loads(cuerpo)["nombre"] == "hoja.jpg"
    estado, _, cuerpo = _pedir(url + "/api/fotos", "POST", b"no soy una imagen", {"X-Nombre": "x.png"})
    assert estado == 400 and "imagen" in json.loads(cuerpo)["error"]
    assert _pedir(url + "/api/nada")[0] == 404
    assert _pedir(url + "/static/../serve.py")[0] == 404
