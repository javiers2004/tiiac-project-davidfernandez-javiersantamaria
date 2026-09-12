"""Tests de forma: la primera linea de defensa del codigo de ML.

Un fallo de formas no da error: da un modelo que entrena sobre basura. Estos
tests son baratos y detectan la mayoria.
"""

import torch

from src.models.factory import construir_modelo

CFG = {"modelo": {"nombre": "mi_modelo"}}


def test_forward_devuelve_la_forma_esperada():
    modelo = construir_modelo(CFG)
    x = torch.randn(4, 32)
    assert modelo(x).shape == (4, 10)


def test_el_batch_no_se_mezcla():
    """Si el modelo mezcla el eje de batch, dos entradas distintas dan la
    misma salida. Es el fallo silencioso mas caro que existe."""
    modelo = construir_modelo(CFG).eval()
    x = torch.randn(8, 32)
    with torch.no_grad():
        juntas = modelo(x)
        sueltas = torch.cat([modelo(x[i:i + 1]) for i in range(8)])
    assert torch.allclose(juntas, sueltas, atol=1e-5)


def test_los_gradientes_llegan_a_todos_los_parametros():
    """Un parametro con gradiente None esta desconectado del grafo: existe,
    ocupa memoria y no aprende."""
    modelo = construir_modelo(CFG)
    modelo(torch.randn(4, 32)).sum().backward()
    sin_gradiente = [n for n, p in modelo.named_parameters()
                     if p.requires_grad and p.grad is None]
    assert not sin_gradiente, f"parametros desconectados: {sin_gradiente}"
