"""Tests de la fachada que consume la aplicacion.

Hasta ahora api.py no tenia ninguno: los de objetivos.py y feature_viz.py
cubren el motor, pero la capa que de verdad va a usar la interfaz estaba sin
red. Estos empiezan por el mapa de activacion.

No usan los .tar: los pesos estan en .gitignore y en CI no existen. Un
resnet18 sin entrenar basta para comprobar la mecanica, porque lo que se
prueba es la forma y el rango del mapa, no que senale hojas.
"""

import pytest
import torch
from torchvision.models import resnet18

from src.interpretability.api import (
    Activacion,
    _activacion_de,
    _preparar,
    mapa_activacion,
    superponer_mapa,
)
from src.interpretability.feature_viz import cargar_config
from src.interpretability.api import CONFIG
from src.utils.seed import fijar_seed

CAPA = "layer3.1.conv2"      # pre-ReLU: tiene valores negativos a proposito


@pytest.fixture(scope="module")
def cfg():
    c = cargar_config(CONFIG)
    c["modelo"] = {**c["modelo"], "n_clases": 10}
    return c


@pytest.fixture(scope="module")
def modelo():
    """Un resnet18 sin entrenar, pero SIEMPRE el mismo.

    Sin fijar la seed, torch siembra su generador global con entropia del
    sistema al arrancar el proceso y los tests serian flaky sin motivo.
    """
    fijar_seed(0)
    m = resnet18(num_classes=10)
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def foto(cfg):
    tam = int(cfg["imagen"]["tamano"])
    return torch.rand(1, 3, tam, tam, generator=torch.Generator().manual_seed(1))


def test_el_mapa_tiene_el_tamano_de_la_foto(foto, modelo, cfg):
    tam = int(cfg["imagen"]["tamano"])
    mapa = mapa_activacion(foto, Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=cfg)
    assert mapa.shape == (tam, tam)
    assert mapa.min() >= 0.0 and mapa.max() <= 1.0


def test_el_mapa_no_es_plano(foto, modelo, cfg):
    """Un mapa constante significa que el hook no capturo lo que creemos, o
    que se esta mapeando el canal equivocado. Se pintaria liso y parece un
    resultado valido."""
    mapa = mapa_activacion(foto, Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=cfg)
    assert mapa.std() > 1e-4


def test_el_maximo_cae_donde_mas_activa_el_canal(foto, modelo, cfg):
    """La propiedad que importa: el mapa tiene que senalar la misma posicion
    que el maximo de la activacion cruda. Si se cambian los ejes al estirar,
    el mapa sale transpuesto y no da ningun error."""
    canal = 7
    act = _activacion_de(_preparar(foto, cfg), CAPA, modelo)[0, canal]
    alto, ancho = act.shape
    plano = int(act.argmax())
    fila_cruda, col_cruda = plano // ancho, plano % ancho

    mapa = mapa_activacion(foto, Activacion(CAPA, canal, 0.0),
                           modelo=modelo, cfg=cfg)
    tam = mapa.shape[0]
    plano_m = int(mapa.argmax())
    fila = plano_m // tam * alto // tam
    col = plano_m % tam * ancho // tam

    assert abs(fila - fila_cruda) <= 1 and abs(col - col_cruda) <= 1


def test_una_capa_sin_extension_espacial_falla(foto, modelo, cfg):
    with pytest.raises(ValueError, match="extension espacial"):
        mapa_activacion(foto, Activacion("fc", 3, 0.0), modelo=modelo, cfg=cfg)


def test_un_canal_inexistente_falla(foto, modelo, cfg):
    with pytest.raises(IndexError, match="no existe"):
        mapa_activacion(foto, Activacion(CAPA, 9999, 0.0), modelo=modelo, cfg=cfg)


def test_un_canal_apagado_da_ceros_y_no_NaN(foto, modelo, cfg):
    """Normalizar dividiendo por el maximo revienta si el canal esta a cero.
    Un mapa de NaN se pinta en negro y parece una respuesta."""
    import copy
    m = copy.deepcopy(modelo)
    m.layer3[1].conv2.register_forward_hook(lambda _m, _e, s: s * 0)
    mapa = mapa_activacion(foto, Activacion(CAPA, 0, 0.0), modelo=m, cfg=cfg)
    assert not torch.isnan(mapa).any()
    assert float(mapa.max()) == 0.0


def test_sin_recortar_el_cero_queda_en_medio(foto, modelo, cfg):
    """Con recortar_negativos=False la escala es simetrica: 0.5 es la
    ausencia de respuesta, y las zonas que inhiben bajan de ahi."""
    mapa = mapa_activacion(foto, Activacion(CAPA, 0, 0.0), modelo=modelo,
                           cfg=cfg, recortar_negativos=False)
    assert mapa.min() < 0.5 < mapa.max()


def test_superponer_devuelve_una_imagen_valida(foto, modelo, cfg):
    tam = int(cfg["imagen"]["tamano"])
    mapa = mapa_activacion(foto, Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=cfg)
    sobre = superponer_mapa(foto, mapa, cfg=cfg)
    assert sobre.shape == (3, tam, tam)
    assert sobre.min() >= 0.0 and sobre.max() <= 1.0


def test_el_hook_no_se_queda_puesto(foto, modelo, cfg):
    """Un hook olvidado se dispara en todas las llamadas siguientes y captura
    activaciones que nadie pidio. _activacion_de lo quita en un finally."""
    modulo = modelo.layer3[1].conv2
    antes = len(modulo._forward_hooks)
    mapa_activacion(foto, Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=cfg)
    assert len(modulo._forward_hooks) == antes
