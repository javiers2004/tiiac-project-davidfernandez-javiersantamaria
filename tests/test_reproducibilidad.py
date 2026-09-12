"""La misma seed tiene que dar el mismo numero. Si este test falla, cualquier
comparacion entre experimentos del proyecto carece de valor."""

import torch

from src.models.factory import construir_modelo
from src.utils.seed import fijar_seed

CFG = {"modelo": {"nombre": "mi_modelo"}}


def test_la_misma_seed_da_los_mismos_pesos():
    fijar_seed(42)
    a = construir_modelo(CFG)
    fijar_seed(42)
    b = construir_modelo(CFG)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_seeds_distintas_dan_pesos_distintos():
    fijar_seed(42)
    a = construir_modelo(CFG)
    fijar_seed(1234)
    b = construir_modelo(CFG)
    assert any(not torch.equal(pa, pb)
               for pa, pb in zip(a.parameters(), b.parameters()))
