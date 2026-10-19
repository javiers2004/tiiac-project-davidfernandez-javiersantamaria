"""Tests de los objetivos del articulo: Neuron, Channel y Layer.

Casi todo lo que puede salir mal aqui sale mal EN SILENCIO: un eje cambiado o
un centro mal redondeado no dan error, dan otra neurona y una imagen que
parece perfectamente razonable. De ahi que estos tests comprueben valores
concretos y no solo formas.
"""

import pytest
import torch
from torchvision.models import resnet18

from src.interpretability.objetivos import (
    Canal,
    Capa,
    FormaCapa,
    Neurona,
    construir_objetivo,
    inspeccionar_capa,
    resolver_modulo,
)


# --- Channel: layer_n[:, :, z] -------------------------------------------

def test_canal_promedia_el_mapa_entero():
    act = torch.arange(2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    assert torch.allclose(Canal(1)(act), act[0, 1].mean())


def test_canal_en_una_capa_lineal_es_el_logit_de_la_clase():
    """En fc no hay extension espacial, asi que Channel se reduce al logit.
    Es el objetivo Class Logits del articulo, gratis."""
    act = torch.randn(1, 10)
    assert torch.allclose(Canal(3)(act), act[0, 3])


# --- Neuron: layer_n[x, y, z] --------------------------------------------

def test_x_es_la_fila_e_y_es_la_columna():
    """El fallo silencioso numero uno.

    lucid indexa su 'x' sobre shape[1], que en NHWC es el ALTO. Si al traducir
    a NCHW se cambian los ejes no hay error ni forma invalida: se optimiza la
    neurona traspuesta. El mapa es no cuadrado para que el cambio se note.
    """
    act = torch.zeros(1, 2, 3, 5)      # (N, C=2, H=3, W=5)
    act[0, 1, 2, 4] = 7.0              # canal 1, fila 2, columna 4
    assert Neurona(1, x=2, y=4)(act).item() == 7.0
    assert Neurona(1, x=1, y=2)(act).item() == 0.0


def test_el_centro_por_defecto_coincide_con_lucid():
    """lucid: shape[1] // 2, shape[2] // 2. Con lado par eso cae en la esquina
    inferior derecha del 2x2 central, que es lo que dice su docstring."""
    assert Neurona(0).posicion(7, 7) == (3, 3)
    assert Neurona(0).posicion(8, 8) == (4, 4)
    assert Neurona(0).posicion(3, 5) == (1, 2)


def test_neurona_toma_una_sola_posicion_no_la_media():
    act = torch.zeros(1, 1, 3, 3)
    act[0, 0, 1, 1] = 9.0
    assert Neurona(0, x=1, y=1)(act).item() == 9.0
    assert Canal(0)(act).item() == pytest.approx(1.0)   # 9 / 9 posiciones


def test_canal_y_neurona_coinciden_si_el_mapa_es_1x1():
    """Sin extension espacial que promediar, los dos objetivos son el mismo.
    Es a donde tiende layer4 cuando la imagen es pequena, y explica por que
    en las capas profundas las dos visualizaciones se parecen tanto."""
    act = torch.randn(1, 4, 1, 1)
    for z in range(4):
        assert torch.allclose(Canal(z)(act), Neurona(z)(act))


def test_neurona_sin_extension_espacial_falla_con_mensaje_util():
    with pytest.raises(ValueError, match="canal"):
        Neurona(0)(torch.randn(1, 10))


# --- Layer (DeepDream): layer_n[:, :, :]^2 -------------------------------

def test_capa_es_la_media_del_cuadrado_de_toda_la_capa():
    """Todos los canales y todas las posiciones, no un canal concreto."""
    act = torch.arange(2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    assert torch.allclose(Capa()(act), act.pow(2).mean())


def test_capa_premia_concentrar_la_activacion():
    """Lo que el cuadrado aporta frente a una media a secas.

    Las dos capas reparten la MISMA activacion total; Channel sobre ellas da
    lo mismo, Layer no. Si alguien quitase el cuadrado, esto lo detecta.
    """
    concentrada = torch.tensor([4.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1)
    repartida = torch.ones(1, 4, 1, 1)
    assert concentrada.mean() == repartida.mean()
    assert Capa()(concentrada).item() == 4.0
    assert Capa()(repartida).item() == 1.0


def test_capa_cuenta_las_activaciones_negativas():
    """En una conv pre-ReLU, hundir una unidad puntua igual que subirla."""
    act = torch.randn(1, 3, 4, 4)
    assert torch.allclose(Capa()(act), Capa()(-act))


# --- Rangos ---------------------------------------------------------------

def test_un_canal_inexistente_falla():
    with pytest.raises(IndexError, match="no existe"):
        Canal(99)(torch.randn(1, 10, 4, 4))


@pytest.mark.parametrize("x,y", [(99, 0), (0, 99), (-1, 0)])
def test_una_posicion_fuera_del_mapa_falla(x, y):
    with pytest.raises(IndexError, match="fuera del mapa"):
        Neurona(0, x=x, y=y)(torch.randn(1, 4, 8, 8))


# --- Construccion desde la configuracion ----------------------------------

def test_construir_objetivo_lee_el_tipo():
    assert construir_objetivo({"tipo": "canal", "canal": 5}) == Canal(5)
    assert construir_objetivo(
        {"tipo": "neurona", "canal": 5, "x": 1, "y": 2}) == Neurona(5, 1, 2)


def test_capa_no_necesita_canal():
    """Layer no elige unidad: el YAML no tiene por que traer 'canal'."""
    assert construir_objetivo({"tipo": "capa"}) == Capa()


def test_neurona_sin_x_ni_y_queda_en_el_centro():
    assert construir_objetivo({"tipo": "neurona", "canal": 0}) == Neurona(0, None, None)


def test_un_tipo_desconocido_falla_y_dice_las_opciones():
    with pytest.raises(ValueError, match="canal, neurona"):
        construir_objetivo({"tipo": "deepdream", "canal": 0})


# --- Metadatos que necesita la aplicacion ---------------------------------

def test_inspeccionar_capa_da_las_formas_reales():
    modelo = resnet18(num_classes=10).eval()
    assert inspeccionar_capa(modelo, "layer4", 224) == FormaCapa("layer4", 512, 7, 7)
    assert inspeccionar_capa(modelo, "layer1", 224) == FormaCapa("layer1", 64, 56, 56)


def test_los_mapas_encogen_con_la_entrada():
    """Los rangos que la interfaz puede ofrecer dependen del tamano de imagen,
    no solo de la capa."""
    modelo = resnet18(num_classes=10).eval()
    assert inspeccionar_capa(modelo, "layer4", 224).alto == 7
    assert inspeccionar_capa(modelo, "layer4", 128).alto == 4


def test_una_capa_lineal_se_declara_no_espacial():
    forma = inspeccionar_capa(resnet18(num_classes=10).eval(), "fc", 224)
    assert not forma.es_espacial
    assert forma.centro is None


def test_resolver_modulo_navega_las_rutas_con_indices():
    modelo = resnet18(num_classes=10)
    assert resolver_modulo(modelo, "fc") is modelo.fc
    assert resolver_modulo(modelo, "layer3.1.conv2") is modelo.layer3[1].conv2
