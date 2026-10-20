"""Tests de la visualizacion por maximizacion de activacion.

Lo que de verdad hay que blindar aqui es que se optimiza LA IMAGEN y no los
pesos. Si los pesos se mueven, la imagen resultante no dice nada del modelo
entrenado: dice algo de un modelo que acabamos de estropear, y el fallo es
silencioso.

No usan los .tar: los pesos estan en .gitignore y en CI no existen. Un
resnet18 sin entrenar basta para comprobar la mecanica.
"""

import copy

import pytest
import torch
from torchvision.models import resnet18

from src.interpretability.feature_viz import (
    ImagenFourier,
    UnidadMuerta,
    alineacion,
    aplicar_robustez,
    maximizar_activacion,
    nombre_salida,
)
from src.interpretability.objetivos import resolver_modulo
from src.utils.seed import fijar_seed

N_CLASES = 10
TAM = 64

CFG = {
    "experimento": {"nombre": "test", "seed": 0},
    "objetivo": {"tipo": "canal", "capa": "fc", "canal": 3},
    "imagen": {"tamano": TAM, "decay_power": 1.0, "sd_inicial": 0.01},
    "optimizacion": {"learning_rate": 0.05, "pasos": 4},
    "regularizacion": {"tv": 0.0, "l2": 0.0},
    "robustez": {"pad": 4, "jitter": 4, "jitter_final": 2,
                 "escalas": [0.95, 1.0, 1.05], "rotaciones": [-5, 0, 5]},
    "normalizacion": {"media": [0.485, 0.456, 0.406],
                      "desviacion": [0.229, 0.224, 0.225]},
}


def _modelo_congelado():
    """Un resnet18 sin entrenar, pero SIEMPRE el mismo.

    Sin fijar la seed, torch siembra su generador global con entropia del
    sistema al arrancar el proceso, asi que cada ejecucion de pytest probaria
    un modelo distinto y los tests serian flaky sin motivo aparente.
    """
    fijar_seed(0)
    modelo = resnet18(num_classes=N_CLASES)
    modelo.eval()
    modelo.requires_grad_(False)
    return modelo


def test_la_imagen_sale_en_el_rango_valido():
    """La sigmoide final tiene que dejar la imagen en [0,1]: si se sale, lo
    que se guarda en el PNG esta recortado y no es lo que se optimizo."""
    imagen = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0))()
    assert imagen.shape == (1, 3, TAM, TAM)
    assert imagen.min() >= 0.0 and imagen.max() <= 1.0


def test_un_lote_da_varias_imagenes_distintas():
    """Cada imagen del lote parte de su propio ruido. Si compartieran el
    ruido, una interpolacion empezaria con todas iguales."""
    imagenes = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0), lote=3)()
    assert imagenes.shape == (3, 3, TAM, TAM)
    assert not torch.allclose(imagenes[0], imagenes[1])


def test_lote_uno_es_la_imagen_de_siempre():
    """Admitir lotes no puede cambiar las visualizaciones que ya existian."""
    a = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0))()
    b = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0), lote=1)()
    assert torch.equal(a, b)


def test_la_robustez_conserva_el_tamano():
    """Las transformaciones recortan al tamano de partida. Si no, el modelo
    recibe una entrada de otro tamano y la comparacion entre pasos se rompe."""
    x = torch.rand(1, 3, TAM, TAM)
    g = torch.Generator().manual_seed(0)
    assert aplicar_robustez(x, CFG["robustez"], g).shape == x.shape


def test_la_robustez_transforma_de_verdad():
    """Con la misma imagen, dos pasadas tienen que diferir: si no, no hay
    aleatoriedad y la robustez no esta haciendo nada."""
    x = torch.rand(1, 3, TAM, TAM)
    g = torch.Generator().manual_seed(0)
    a = aplicar_robustez(x, CFG["robustez"], g)
    b = aplicar_robustez(x, CFG["robustez"], g)
    assert not torch.allclose(a, b)


def test_resolver_modulo_navega_las_rutas_con_indices():
    modelo = _modelo_congelado()
    assert resolver_modulo(modelo, "fc") is modelo.fc
    assert resolver_modulo(modelo, "layer3.1.conv2") is modelo.layer3[1].conv2


def test_la_robustez_transforma_igual_todo_el_lote():
    """Si cada imagen recibiera un encuadre distinto, las de una
    interpolacion se desalinearian entre si."""
    x = torch.rand(1, 3, TAM, TAM).repeat(2, 1, 1, 1)
    salida = aplicar_robustez(x, CFG["robustez"], torch.Generator().manual_seed(0))
    assert torch.equal(salida[0], salida[1])


def test_el_bucle_devuelve_el_lote_entero():
    cfg = copy.deepcopy(CFG)
    cfg["imagen"]["lote"] = 3
    imagenes, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())
    assert imagenes.shape == (3, 3, TAM, TAM)
    assert len(historial) == CFG["optimizacion"]["pasos"]


def test_los_pesos_no_se_mueven():
    """El test que importa: optimizamos la imagen, no el modelo."""
    modelo = _modelo_congelado()
    antes = copy.deepcopy(modelo.state_dict())

    maximizar_activacion(CFG, modelo=modelo)

    despues = modelo.state_dict()
    movidos = [k for k in antes if not torch.equal(antes[k], despues[k])]
    assert not movidos, f"estos pesos han cambiado: {movidos}"


def test_la_activacion_sube():
    """Si la activacion no sube, el gradiente no esta llegando a la imagen.

    Se comparan medias y no los extremos: cada paso mide la activacion bajo
    una transformacion aleatoria distinta, asi que un paso suelto puede bajar
    sin que la optimizacion vaya mal. Comparar historial[-1] con historial[0]
    hace el test flaky.
    """
    cfg = copy.deepcopy(CFG)
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    assert len(historial) == 12
    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_deepdream_sube_la_energia_de_la_capa():
    """El mismo bucle, con el objetivo Layer sobre una capa intermedia."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "capa", "capa": "layer2"}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_la_direccion_sube():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "direccion", "capa": "layer3.1.conv2", "direccion": 0}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_el_logit_de_la_clase_sube():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "logit", "capa": "fc", "canal": 3}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_la_probabilidad_de_la_clase_sube():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "probabilidad", "capa": "fc", "canal": 3}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    assert all(0.0 <= p <= 1.0 for p in historial)
    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:.4f} a {ultimos:.4f}"


def test_logit_y_canal_en_fc_dan_la_misma_imagen():
    """Class Logits es Channel sobre fc; la imagen tiene que ser identica."""
    modelo = _modelo_congelado()
    logit_cfg = copy.deepcopy(CFG)
    logit_cfg["objetivo"]["tipo"] = "logit"
    a, _ = maximizar_activacion(CFG, modelo=modelo)
    b, _ = maximizar_activacion(logit_cfg, modelo=modelo)
    assert torch.allclose(a, b)


def test_un_canal_inexistente_falla_pronto():
    """Sin la comprobacion, un indice fuera de rango revienta con un error de
    indexado de torch mucho mas abajo y cuesta de leer."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"]["canal"] = N_CLASES + 5
    with pytest.raises(IndexError, match="no existe"):
        maximizar_activacion(cfg, modelo=_modelo_congelado())


def test_una_unidad_muerta_se_detecta_en_el_primer_paso():
    """Una unidad que un ReLU deja en cero no propaga gradiente: el bucle
    devolveria el ruido inicial sin quejarse. Mas de la mitad de las
    posiciones centrales de layer4 estan asi al arrancar, asi que esto no es
    un caso raro."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "neurona", "capa": "layer4", "canal": 0}

    modelo = _modelo_congelado()
    # s * 0, no zeros_like: una unidad muerta de verdad SIGUE en el grafo, lo
    # que pasa es que su gradiente es cero. Con zeros_like el tensor queda
    # desconectado y backward falla por otro motivo, que no es el que se
    # quiere probar.
    modelo.layer4.register_forward_hook(lambda _m, _e, s: s * 0)

    with pytest.raises(UnidadMuerta, match="no propaga gradiente"):
        maximizar_activacion(cfg, modelo=modelo)


# --- Alineacion -----------------------------------------------------------

def test_la_alineacion_es_cero_si_el_lote_es_igual():
    act = torch.randn(1, 4, 5, 5).repeat(3, 1, 1, 1)
    assert alineacion(act, 2.0, 4).item() == 0.0


def test_la_alineacion_de_dos_imagenes():
    a, b = torch.zeros(1, 2, 3, 3), torch.ones(1, 2, 3, 3)
    assert alineacion(torch.cat([a, b]), 2.0, 4).item() == pytest.approx(0.5)


def test_las_parejas_lejanas_pesan_menos():
    """Con tres imagenes 0, 0, 1: la pareja (1,2) pesa 1/2 y la (0,2), que
    estan a distancia 2, pesa 1/4. Si el decaimiento no se aplicase, o se
    aplicase al reves, el total seria otro."""
    act = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1, 1)
    assert alineacion(act, 2.0, 4).item() == pytest.approx(0.5 + 0.25)
    assert alineacion(act, 2.0, 1).item() == pytest.approx(0.5)


def test_la_alineacion_acerca_las_imagenes_del_lote():
    """El efecto que se busca: con alineacion fuerte, las imagenes de la fila
    acaban mas parecidas entre si que sin ella."""
    def separacion(peso):
        cfg = copy.deepcopy(CFG)
        cfg["imagen"]["lote"] = 3
        cfg["optimizacion"]["pasos"] = 12
        cfg["regularizacion"]["alineacion"] = {"peso": peso, "capa": "layer1",
                                               "decaimiento": 2.0, "alcance": 4}
        imagenes, _ = maximizar_activacion(cfg, modelo=_modelo_congelado())
        return (imagenes[1:] - imagenes[:-1]).abs().mean().item()

    assert separacion(1000.0) < separacion(0.0)


def test_la_alineacion_no_tapa_una_unidad_muerta():
    """La alineacion tambien mueve la imagen. Si la comprobacion mirase el
    gradiente total, una unidad muerta pasaria desapercibida."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "canal", "capa": "layer4", "canal": 0}
    cfg["imagen"]["lote"] = 2
    cfg["regularizacion"]["alineacion"] = {"peso": 1.0, "capa": "layer1",
                                           "decaimiento": 2.0, "alcance": 4}
    modelo = _modelo_congelado()
    modelo.layer4.register_forward_hook(lambda _m, _e, s: s * 0)
    with pytest.raises(UnidadMuerta):
        maximizar_activacion(cfg, modelo=modelo)


def test_los_hooks_se_quitan_al_terminar():
    """Si quedase uno enganchado, la siguiente llamada con el mismo modelo
    (la app lo reutiliza) capturaria de mas y gastaria memoria."""
    cfg = copy.deepcopy(CFG)
    cfg["imagen"]["lote"] = 2
    cfg["regularizacion"]["alineacion"] = {"peso": 1.0, "capa": "layer1",
                                           "decaimiento": 2.0, "alcance": 4}
    modelo = _modelo_congelado()
    maximizar_activacion(cfg, modelo=modelo)
    assert not modelo.fc._forward_hooks and not modelo.layer1._forward_hooks


def test_el_nombre_del_fichero_distingue_los_dos_objetivos():
    """Channel y Neuron sobre la misma capa y canal no pueden acabar en el
    mismo PNG, o uno pisa al otro sin avisar."""
    base = {"capa": "layer2", "canal": 42}
    canal = nombre_salida({**base, "tipo": "canal"})
    neurona = nombre_salida({**base, "tipo": "neurona", "x": None, "y": None})
    assert canal != neurona


def test_el_nombre_del_deepdream_no_depende_del_canal():
    """Layer no elige canal: lo que quede en el YAML no puede cambiar el
    nombre del PNG, o el mismo DeepDream se guardaria dos veces."""
    a = nombre_salida({"tipo": "capa", "capa": "layer3", "canal": 0})
    b = nombre_salida({"tipo": "capa", "capa": "layer3", "canal": 42})
    assert a == b == "layer3_capa"


def test_el_nombre_de_la_direccion_es_su_semilla():
    """Dos direcciones distintas no pueden pisarse el PNG, y el canal que
    quede en el YAML no puede cambiar el nombre de la misma direccion."""
    a = nombre_salida({"tipo": "direccion", "capa": "layer4", "direccion": 1, "canal": 0})
    b = nombre_salida({"tipo": "direccion", "capa": "layer4", "direccion": 1, "canal": 9})
    c = nombre_salida({"tipo": "direccion", "capa": "layer4", "direccion": 2, "canal": 0})
    assert a == b == "layer4_direccion_s1"
    assert a != c


def test_el_nombre_de_la_suma_lleva_todos_los_canales():
    """A+B y A solo no pueden pisarse el PNG."""
    assert nombre_salida({"tipo": "suma", "capa": "layer4", "canales": [12, 40]}) \
        == "layer4_suma_12+40"


def test_la_misma_seed_da_la_misma_imagen():
    """Sin esto, dos ejecuciones dan visualizaciones distintas y no se puede
    comparar una neurona consigo misma entre commits.

    El modelo se reutiliza a proposito: en uso real sale siempre del mismo
    checkpoint, asi que lo que se comprueba es que la aleatoriedad de la
    optimizacion (ruido inicial y transformaciones) esta bajo la seed.
    """
    modelo = _modelo_congelado()
    a, _ = maximizar_activacion(CFG, modelo=modelo)
    b, _ = maximizar_activacion(CFG, modelo=modelo)
    assert torch.allclose(a, b)
