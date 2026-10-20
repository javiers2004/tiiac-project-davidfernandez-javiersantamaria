"""Tests de los objetivos del articulo: Neuron, Channel, Layer, Class Logits,
Class Probability y, de "The Interaction between Neurons", Direction e
Interpolation.

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
    Direccion,
    FormaCapa,
    Interpolacion,
    Logit,
    Neurona,
    Probabilidad,
    Suma,
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
    with pytest.raises(ValueError, match="logit"):
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


# --- Class Logits: pre_softmax[k] ----------------------------------------

def test_logit_es_la_salida_de_la_clase():
    act = torch.randn(1, 10)
    assert torch.allclose(Logit(3)(act), act[0, 3])


def test_logit_coincide_con_canal_sobre_la_salida():
    """Mismo numero que Canal(k) en fc: Logit no cambia la optimizacion de
    clases que ya se hacia, solo la nombra y la protege."""
    act = torch.randn(1, 10)
    for k in range(10):
        assert torch.allclose(Logit(k)(act), Canal(k)(act))


def test_logit_fuera_de_la_salida_falla():
    """Canal(k) sobre layer4 optimiza el canal k de una conv sin quejarse;
    Logit(k) no puede acabar etiquetando eso como una clase."""
    with pytest.raises(ValueError, match="capa de salida"):
        Logit(0)(torch.randn(1, 512, 7, 7))


def test_una_clase_inexistente_falla():
    with pytest.raises(IndexError, match="la clase 10 no existe"):
        Logit(10)(torch.randn(1, 10))
    with pytest.raises(IndexError, match="la clase 10 no existe"):
        Probabilidad(10)(torch.randn(1, 10))


# --- Class Probability: softmax[k] ---------------------------------------

def test_probabilidad_es_la_softmax_de_la_clase():
    act = torch.randn(1, 10)
    assert torch.allclose(Probabilidad(3)(act), act.softmax(dim=-1)[0, 3])


def test_las_probabilidades_suman_uno():
    """Si la softmax se tomase sobre el eje equivocado (el de batch), esto no
    se cumpliria y no habria ningun otro error."""
    act = torch.randn(1, 10)
    total = sum(Probabilidad(k)(act) for k in range(10))
    assert total.item() == pytest.approx(1.0)


def test_probabilidad_sube_sin_tocar_el_logit():
    """El motivo por el que el articulo prefiere los logits.

    Mismo logit de la clase 0 en las dos salidas; en la segunda solo se han
    hundido las demas. Logit no ve diferencia, Probabilidad si: puede subir
    sin que la imagen se parezca mas a la clase.
    """
    antes = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    despues = torch.tensor([[2.0, -3.0, -3.0, -3.0]])
    assert Logit(0)(antes) == Logit(0)(despues)
    assert Probabilidad(0)(despues) > Probabilidad(0)(antes)


def test_probabilidad_fuera_de_la_salida_falla():
    with pytest.raises(ValueError, match="capa de salida"):
        Probabilidad(0)(torch.randn(1, 512, 7, 7))


# --- Direction: v . layer_n[x, y, :] ------------------------------------

def test_direccion_es_una_combinacion_de_canales():
    """Lo que el articulo dice del espacio de activaciones, en un test: una
    direccion es la suma de los canales pesados por v. Si el producto escalar
    se hiciese sobre el eje equivocado no daria error, daria otro numero."""
    act = torch.randn(1, 6, 4, 5)
    d = Direccion(3)
    v = d.vector(6)
    combinacion = sum(v[c] * Canal(c)(act) for c in range(6))
    assert torch.allclose(d(act), combinacion, atol=1e-6)


def test_la_direccion_tiene_norma_uno():
    assert Direccion(0).vector(512).norm().item() == pytest.approx(1.0)


def test_la_misma_semilla_es_la_misma_direccion():
    """Si no, la direccion que se visualiza no es la que se ha guardado."""
    assert torch.equal(Direccion(7).vector(64), Direccion(7).vector(64))
    assert not torch.allclose(Direccion(7).vector(64), Direccion(8).vector(64))


def test_la_direccion_no_toca_el_estado_global_de_torch():
    """Las semillas globales las fija seed.py; un objetivo no puede moverlas."""
    act = torch.randn(1, 8, 3, 3)
    torch.manual_seed(0)
    esperado = torch.rand(1)
    torch.manual_seed(0)
    Direccion(5)(act)
    assert torch.equal(torch.rand(1), esperado)


def test_direccion_funciona_tambien_en_la_salida():
    """En fc es una combinacion de logits."""
    act = torch.randn(1, 10)
    v = Direccion(1).vector(10)
    assert torch.allclose(Direccion(1)(act), (act[0] * v).sum())


# --- Interpolacion: (1 - w_i) A + w_i B ---------------------------------

def test_los_extremos_de_la_fila_son_las_dos_neuronas():
    """La primera imagen solo ve a A y la ultima solo a B. Con los pesos al
    reves la fila saldria invertida, sin ningun error."""
    w = Interpolacion.pesos(5)
    assert w[0] == 0.0 and w[-1] == 1.0
    assert torch.allclose(w, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_cada_imagen_tiene_su_peso():
    """Imagen 0 con A=1 y B=0, imagen 1 con A=0 y B=1, en una fila de 2: la
    imagen 0 solo puntua por A y la 1 solo por B, asi que da 1 de media."""
    act = torch.zeros(2, 2, 3, 3)
    act[0, 0] = 1.0     # imagen 0, canal A
    act[1, 1] = 1.0     # imagen 1, canal B
    assert Interpolacion(0, 1)(act).item() == pytest.approx(1.0)
    # Cambiadas de sitio, cada imagen activa justo el canal que no le toca.
    assert Interpolacion(1, 0)(act).item() == pytest.approx(0.0)


def test_la_interpolacion_de_un_canal_consigo_mismo_es_canal():
    act = torch.randn(4, 3, 5, 5)
    assert torch.allclose(Interpolacion(2, 2)(act), Canal(2)(act))


def test_la_interpolacion_necesita_un_lote():
    with pytest.raises(ValueError, match="imagen.lote"):
        Interpolacion(0, 1)(torch.randn(1, 4, 3, 3))


def test_la_interpolacion_valida_los_canales():
    with pytest.raises(IndexError, match="no existe"):
        Interpolacion(0, 99)(torch.randn(2, 4, 3, 3))


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


def test_direccion_lee_su_semilla():
    assert construir_objetivo({"tipo": "direccion", "direccion": 4}) == Direccion(4)


def test_suma_lee_la_lista_de_canales():
    assert construir_objetivo({"tipo": "suma", "canales": [3, 8]}) == Suma((Canal(3), Canal(8)))


def test_suma_es_la_suma_de_sus_canales():
    act = torch.randn(1, 10, 4, 4)
    suma = construir_objetivo({"tipo": "suma", "canales": [3, 8]})
    assert torch.allclose(suma(act), Canal(3)(act) + Canal(8)(act))


def test_interpolacion_lee_sus_dos_canales():
    ocfg = {"tipo": "interpolacion", "canales": [12, 40]}
    assert construir_objetivo(ocfg) == Interpolacion(12, 40)


def test_la_interpolacion_quiere_exactamente_dos_canales():
    with pytest.raises(ValueError, match="2"):
        construir_objetivo({"tipo": "interpolacion", "canales": [1, 2, 3]})


def test_logit_lee_la_clase_de_canal():
    """En fc el indice z es la clase, igual que con Channel."""
    assert construir_objetivo({"tipo": "logit", "canal": 7}) == Logit(7)
    assert construir_objetivo({"tipo": "probabilidad", "canal": 7}) == Probabilidad(7)


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
