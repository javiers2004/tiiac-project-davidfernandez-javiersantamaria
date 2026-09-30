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
    aplicar_robustez,
    maximizar_activacion,
    resolver_modulo,
)

N_CLASES = 10
TAM = 64

CFG = {
    "experimento": {"nombre": "test", "seed": 0},
    "objetivo": {"capa": "fc", "neurona": 3},
    "imagen": {"tamano": TAM, "decay_power": 1.0, "sd_inicial": 0.01},
    "optimizacion": {"learning_rate": 0.05, "pasos": 4},
    "regularizacion": {"tv": 0.0, "l2": 0.0},
    "robustez": {"pad": 4, "jitter": 4, "jitter_final": 2,
                 "escalas": [0.95, 1.0, 1.05], "rotaciones": [-5, 0, 5]},
    "normalizacion": {"media": [0.485, 0.456, 0.406],
                      "desviacion": [0.229, 0.224, 0.225]},
}


def _modelo_congelado():
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


def test_los_pesos_no_se_mueven():
    """El test que importa: optimizamos la imagen, no el modelo."""
    modelo = _modelo_congelado()
    antes = copy.deepcopy(modelo.state_dict())

    maximizar_activacion(CFG, modelo=modelo)

    despues = modelo.state_dict()
    movidos = [k for k in antes if not torch.equal(antes[k], despues[k])]
    assert not movidos, f"estos pesos han cambiado: {movidos}"


def test_la_activacion_sube():
    """Si la activacion no sube, el gradiente no esta llegando a la imagen."""
    _, historial = maximizar_activacion(CFG, modelo=_modelo_congelado())
    assert len(historial) == CFG["optimizacion"]["pasos"]
    assert historial[-1] > historial[0]


def test_una_neurona_inexistente_falla_pronto():
    """Sin la comprobacion, un indice fuera de rango revienta con un error de
    indexado de torch mucho mas abajo y cuesta de leer."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"]["neurona"] = N_CLASES + 5
    with pytest.raises(IndexError, match="no existe"):
        maximizar_activacion(cfg, modelo=_modelo_congelado())


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
