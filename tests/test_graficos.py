"""Tests de la vista de la red de la interfaz.

La especificacion se construye entera en Python, en un unico sistema de
coordenadas, y de ahi salen los fallos que importan: una celda fuera de su
placa o dos etapas superpuestas NO dan error, dan un dibujo roto que solo se
ve abriendo la aplicacion y mirandola. Por eso estos tests comprueban
geometria y no solo que el diccionario tenga las claves.
"""

import math

import pytest

from src.app.graficos import PARAM_CANAL, PARAM_ETAPA, espec_red

ETAPAS = [
    {"capa": "layer1.1.conv2", "corta": "layer1", "canales": 64,
     "lado": 56, "forma": "64 x 56 x 56"},
    {"capa": "layer2.1.conv2", "corta": "layer2", "canales": 128,
     "lado": 28, "forma": "128 x 28 x 28"},
    {"capa": "layer3.1.conv2", "corta": "layer3", "canales": 256,
     "lado": 14, "forma": "256 x 14 x 14"},
    {"capa": "layer4.1.conv2", "corta": "layer4", "canales": 512,
     "lado": 7, "forma": "512 x 7 x 7"},
    {"capa": "fc", "corta": "fc", "canales": 1081, "lado": 0,
     "forma": "1081 clases"},
]

CONVOLUCIONALES = [("layer1.1.conv2", 64), ("layer2.1.conv2", 128),
                   ("layer3.1.conv2", 256), ("layer4.1.conv2", 512)]


def partes(spec):
    """Las tres capas del grafico: barras, celdas y rotulos."""
    return [c["data"]["values"] for c in spec["layer"]]


def construir(capa, n):
    return espec_red(ETAPAS, capa, [0.01 * i for i in range(n)],
                     canal=0, destacados={0, 1, 2})


# --------------------------------------------------------------- estructura

def test_hay_una_barra_y_un_rotulo_por_etapa():
    barras, _, rotulos = partes(construir("layer3.1.conv2", 256))
    assert len(barras) == len(ETAPAS)
    assert len(rotulos) == len(ETAPAS)


def test_los_dos_parametros_de_seleccion_estan_declarados():
    spec = construir("layer3.1.conv2", 256)
    nombres = {p["name"] for c in spec["layer"] for p in c.get("params", [])}
    assert nombres == {PARAM_ETAPA, PARAM_CANAL}


@pytest.mark.parametrize("capa,n", CONVOLUCIONALES)
def test_hay_una_celda_por_canal(capa, n):
    _, celdas, _ = partes(construir(capa, n))
    assert len(celdas) == n
    assert {c["canal"] for c in celdas} == set(range(n))


def test_fc_no_dibuja_rejilla():
    """1081 celdas dentro de una etapa no se pueden pulsar: ahi no va rejilla."""
    _, celdas, _ = partes(espec_red(ETAPAS, "fc", None, None, set()))
    assert celdas == []


# ---------------------------------------------------------------- geometria

@pytest.mark.parametrize("capa,n", CONVOLUCIONALES)
def test_las_etapas_no_se_solapan(capa, n):
    barras, _, _ = partes(construir(capa, n))
    for izquierda, derecha in zip(barras, barras[1:]):
        assert izquierda["x1"] <= derecha["x0"]


@pytest.mark.parametrize("capa,n", CONVOLUCIONALES)
def test_la_etapa_abierta_es_la_mas_ancha(capa, n):
    barras, _, _ = partes(construir(capa, n))
    abierta = next(b for b in barras if b["estado"] == "abierta")
    cerradas = [b for b in barras if b["estado"] == "cerrada"]
    ancho = abierta["x1"] - abierta["x0"]
    assert all(ancho > b["x1"] - b["x0"] for b in cerradas)


@pytest.mark.parametrize("capa,n", CONVOLUCIONALES)
def test_las_celdas_caben_dentro_de_su_etapa(capa, n):
    """El fallo que motivo estos tests: la rejilla saliendose de su placa."""
    barras, celdas, _ = partes(construir(capa, n))
    abierta = next(b for b in barras if b["estado"] == "abierta")
    for c in celdas:
        assert abierta["x0"] <= c["x0"] < c["x1"] <= abierta["x1"]
        assert abierta["y0"] <= c["y0"] < c["y1"] <= abierta["y1"]


@pytest.mark.parametrize("capa,n", CONVOLUCIONALES)
def test_las_celdas_no_se_solapan_entre_si(capa, n):
    _, celdas, _ = partes(construir(capa, n))
    for a, b in zip(celdas, celdas[1:]):
        misma_fila = math.isclose(a["y0"], b["y0"])
        assert (a["x1"] <= b["x0"]) if misma_fila else (b["y1"] <= a["y0"])


@pytest.mark.parametrize("capa,n", CONVOLUCIONALES)
def test_la_rejilla_no_deja_filas_cojas(capa, n):
    """Los canales van en potencias de dos: la rejilla tiene que salir exacta."""
    _, celdas, _ = partes(construir(capa, n))
    columnas = sum(1 for c in celdas if math.isclose(c["y0"], celdas[0]["y0"]))
    assert n % columnas == 0


def test_la_altura_de_la_barra_es_el_lado_del_mapa():
    """El embudo tiene que ser honesto: 56, 28, 14 y 7."""
    barras, _, _ = partes(construir("layer1.1.conv2", 64))
    cerradas = {b["capa"]: b for b in barras if b["estado"] == "cerrada"}
    for e in ETAPAS:
        if e["capa"] in cerradas and e["lado"]:
            b = cerradas[e["capa"]]
            assert math.isclose(b["y1"] - b["y0"], e["lado"])


# -------------------------------------------------------------------- marcas

def test_solo_el_canal_elegido_lleva_la_marca_de_elegido():
    _, celdas, _ = partes(espec_red(ETAPAS, "layer3.1.conv2",
                                    [0.0] * 256, canal=7, destacados={1, 2}))
    elegidos = [c["canal"] for c in celdas if c["marca"] == "elegido"]
    destacados = {c["canal"] for c in celdas if c["marca"] == "destacado"}
    assert elegidos == [7]
    assert destacados == {1, 2}


def test_el_canal_elegido_manda_sobre_el_destacado():
    """Si el elegido esta en el top, se pinta como elegido, no como destacado."""
    _, celdas, _ = partes(espec_red(ETAPAS, "layer3.1.conv2",
                                    [0.0] * 256, canal=1, destacados={1, 2}))
    assert celdas[1]["marca"] == "elegido"
