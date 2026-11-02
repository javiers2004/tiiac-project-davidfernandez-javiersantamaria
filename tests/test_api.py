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


# --- Progreso y cancelacion ----------------------------------------------
#
# generar_imagen bloquea entre 14 s y 3 minutos. Sin estas dos cosas, una
# interfaz se queda muda todo ese rato y no hay forma de abortar.

def _cfg_rapida(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c["optimizacion"] = {**c["optimizacion"], "pasos": 6}
    return c


def test_el_callback_se_llama_en_cada_paso(modelo, cfg):
    from src.interpretability.api import generar_imagen
    pasos = []
    generar_imagen(Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=_cfg_rapida(cfg),
                   al_progresar=lambda p, t, a, im: pasos.append((p, t, im.shape)))
    assert [p for p, _, _ in pasos] == list(range(6))
    assert all(t == 6 for _, t, _ in pasos)


def test_la_vista_previa_viene_desconectada_del_grafo(modelo, cfg):
    """Si se pasara el tensor vivo, quien guarde la vista previa mantendria
    el grafo entero en memoria, paso tras paso."""
    from src.interpretability.api import generar_imagen
    vistas = []
    generar_imagen(Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=_cfg_rapida(cfg),
                   al_progresar=lambda p, t, a, im: vistas.append(im))
    assert all(not v.requires_grad and v.grad_fn is None for v in vistas)
    assert all(v.min() >= 0.0 and v.max() <= 1.0 for v in vistas)


def test_devolver_False_corta_la_optimizacion(modelo, cfg):
    from src.interpretability.api import generar_imagen
    import copy
    c = copy.deepcopy(cfg)
    c["optimizacion"] = {**c["optimizacion"], "pasos": 50}
    pasos = []

    def corta(p, t, a, im):
        pasos.append(p)
        return False if p >= 3 else None

    imagen = generar_imagen(Activacion(CAPA, 0, 0.0), modelo=modelo, cfg=c,
                            al_progresar=corta)
    assert pasos == [0, 1, 2, 3]          # para en el 3, no en el 50
    assert imagen.min() >= 0.0 and imagen.max() <= 1.0


def test_sin_callback_todo_sigue_igual(modelo, cfg):
    """El callback es opcional: ningun llamador existente debia cambiar."""
    from src.interpretability.api import generar_imagen
    imagen = generar_imagen(Activacion(CAPA, 0, 0.0), modelo=modelo,
                            cfg=_cfg_rapida(cfg))
    assert imagen.shape[0] == 1 and imagen.shape[1] == 3


# --- Diversidad, a traves de la fachada ----------------------------------
#
# El termino lo cubren los tests de feature_viz.py, pero generar_diversas,
# que es lo que llamara la aplicacion, no tenia ninguno.

def test_generar_diversas_devuelve_un_lote(modelo, cfg):
    from src.interpretability.api import generar_diversas
    fotos = generar_diversas(Activacion(CAPA, 0, 0.0), imagenes=3,
                             modelo=modelo, cfg=_cfg_rapida(cfg))
    assert fotos.shape[0] == 3
    assert fotos.min() >= 0.0 and fotos.max() <= 1.0


def test_generar_diversas_da_imagenes_distintas(modelo, cfg):
    """Si salen iguales, el termino de diversidad no esta llegando y la
    funcion no sirve para lo que dice servir."""
    from src.interpretability.api import generar_diversas
    fotos = generar_diversas(Activacion(CAPA, 0, 0.0), imagenes=2, peso=50.0,
                             modelo=modelo, cfg=_cfg_rapida(cfg))
    assert not torch.allclose(fotos[0], fotos[1])


# --- Seguimiento: sin clave de W&B tiene que funcionar igual ------------------

def _sin_clave(monkeypatch, tmp_path):
    """Un entorno sin WANDB_API_KEY y sin ~/.netrc, como el de quien clona."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NETRC", str(tmp_path / "netrc"))


def test_sin_clave_de_wandb_no_se_abre_run(monkeypatch, tmp_path, capsys):
    """wandb.init sin clave pregunta por terminal y acaba reventando: quien
    solo quiere reproducir una imagen se quedaba sin ella."""
    import wandb
    from src.utils.tracking import abrir_run

    def no_llamar(**_):
        raise AssertionError("no se puede llamar a wandb.init sin clave")

    _sin_clave(monkeypatch, tmp_path)
    monkeypatch.setattr(wandb, "init", no_llamar)
    cfg = {"experimento": {"nombre": "t"}, "wandb": {"modo": "online"}}
    assert abrir_run(cfg) is None
    assert "WANDB_API_KEY" in capsys.readouterr().out


def test_con_clave_de_wandb_se_abre_run(monkeypatch, tmp_path):
    import wandb
    from src.utils.tracking import abrir_run

    _sin_clave(monkeypatch, tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setattr(wandb, "init", lambda **_: "run")
    assert abrir_run({"experimento": {"nombre": "t"}, "wandb": {"modo": "online"}}) == "run"


def test_el_modo_offline_no_necesita_clave(monkeypatch, tmp_path):
    """offline guarda en local para sincronizar despues: la clave no hace falta."""
    import wandb
    from src.utils.tracking import abrir_run

    _sin_clave(monkeypatch, tmp_path)
    monkeypatch.setattr(wandb, "init", lambda **_: "run")
    assert abrir_run({"experimento": {"nombre": "t"}, "wandb": {"modo": "offline"}}) == "run"

