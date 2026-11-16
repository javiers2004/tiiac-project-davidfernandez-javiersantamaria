"""Especificacion Vega-Lite de la vista de la red, pulsable en dos niveles.

Es una funcion pura: construye el diccionario del grafico y nada mas. No
importa streamlit ni torch, asi que se puede probar sin levantar la
aplicacion ni cargar el checkpoint.

La vista tiene dos niveles. En reposo, las cinco etapas de la red son barras
cuya altura es el lado de su mapa de activacion: sale el embudo de un
ResNet, la resolucion cayendo mientras los canales se multiplican. Al pulsar
una etapa, esa se ensancha y dentro aparece la rejilla de TODOS sus canales,
coloreados por lo que los activa la foto; pulsando una celda se elige la
neurona a optimizar.

Quien lo pinta es st.vega_lite_chart(..., on_select="rerun"), que devuelve
lo seleccionado en event.selection[<nombre del parametro>]. Hay dos
parametros, uno por nivel: PARAM_ETAPA y PARAM_CANAL.

TODO se dibuja en un unico sistema de coordenadas continuo, calculado aqui
en Python. Es a proposito: con un eje de bandas no hay forma de meter una
rejilla dentro de una banda, y mezclar coordenadas de escala con tamanos en
pixeles hace que el dibujo se rompa en cuanto cambia el ancho de la ventana.
"""

from __future__ import annotations

import math
from typing import Any

PARAM_ETAPA = "etapa"
PARAM_CANAL = "canal"

# Lienzo en unidades propias. El ancho real lo pone la ventana; lo que
# importa es que todas las piezas se midan en las MISMAS unidades.
_ANCHO = 1000.0
_Y_MIN, _Y_MAX = -38.0, 30.0
_Y_ROTULO = -34.0

# Reparto del ancho entre etapas.
_HUECO = 14.0            # entre etapas
_ABIERTA = 420.0         # ancho de la etapa abierta

# La etapa abierta es una placa de este alto, y la rejilla vive dentro con
# un margen.
_PLACA = 56.0
_MARGEN_X, _MARGEN_Y = 12.0, 4.0

# fc no tiene extension espacial: se dibuja como una banda fina.
_ALTO_FC = 5.0

# Alto del area de dibujo, en pixeles. Hace falta para estimar la proporcion
# de una celda: el eje x reparte _ANCHO unidades sobre el ancho de la
# ventana y el eje y reparte (_Y_MAX - _Y_MIN) sobre estos pixeles, asi que
# una unidad de x y una de y NO miden lo mismo en pantalla.
_ALTO_PX = 230.0
_ANCHO_PX_NOMINAL = 900.0   # ancho tipico; solo afecta a cuantas columnas salen

_MAX_COLUMNAS = 48


def _columnas(n: int, ancho_u: float, alto_u: float) -> int:
    """Cuantas columnas dejan las celdas lo mas cuadradas posible en pantalla.

    Args:
        n: numero de canales.
        ancho_u: ancho disponible, en unidades del eje x.
        alto_u: alto disponible, en unidades del eje y.

    Returns:
        El numero de columnas, al menos 1.
    """
    px_x = ancho_u * (_ANCHO_PX_NOMINAL / _ANCHO)
    px_y = alto_u * (_ALTO_PX / (_Y_MAX - _Y_MIN))
    if px_y <= 0:
        return min(n, _MAX_COLUMNAS)
    ideal = round(math.sqrt(max(n, 1) * px_x / px_y))
    ideal = max(1, min(ideal, _MAX_COLUMNAS, max(n, 1)))

    # Los canales de un ResNet van en potencias de dos, asi que siempre hay
    # un divisor exacto razonablemente cerca del ideal. Usarlo evita que la
    # ultima fila quede coja. Se buscan TODOS los divisores, no solo los
    # pegados al ideal: para 256 canales el ideal son 24 columnas y el
    # divisor mas cercano es 32, a ocho de distancia.
    divisores = [c for c in range(1, min(_MAX_COLUMNAS, n) + 1) if n % c == 0]
    # En empate gana el mayor: mas columnas y menos filas aprovechan mejor
    # una placa que es mas ancha que alta.
    mejor = min(divisores, key=lambda c: (abs(c - ideal), -c))
    return mejor if abs(mejor - ideal) <= ideal * 0.5 else ideal


def _ranuras(etapas: list[dict[str, Any]], capa: str) -> list[tuple[float, float]]:
    """El tramo de x que ocupa cada etapa, con la elegida ensanchada."""
    n = len(etapas)
    abiertas = sum(1 for e in etapas if e["capa"] == capa)
    resto = _ANCHO - _HUECO * (n - 1) - _ABIERTA * abiertas
    estrecha = resto / max(n - abiertas, 1)

    ranuras, x = [], 0.0
    for e in etapas:
        w = _ABIERTA if e["capa"] == capa else estrecha
        ranuras.append((x, x + w))
        x += w + _HUECO
    return ranuras


def espec_red(etapas: list[dict[str, Any]], capa: str,
              valores: list[float] | None = None, canal: int | None = None,
              destacados: set[int] | None = None) -> dict[str, Any]:
    """La red por etapas, con la etapa abierta ensenando sus canales dentro.

    Args:
        etapas: una entrada por capa seleccionable, con las claves 'capa'
            (la ruta completa, p. ej. 'layer3.1.conv2'), 'corta' (el
            rotulo), 'canales', 'lado' (0 si no tiene extension espacial) y
            'forma' (el texto del tooltip).
        capa: la ruta de la etapa abierta.
        valores: activacion media de cada canal de esa etapa, en orden de
            indice. None si la etapa no tiene rejilla que ensenar.
        canal: el canal elegido dentro de la etapa abierta.
        destacados: canales del top del ranking, que van con borde.

    Returns:
        La especificacion Vega-Lite, lista para st.vega_lite_chart.
    """
    destacados = destacados or set()
    ranuras = _ranuras(etapas, capa)

    barras, rotulos, celdas = [], [], []
    for (x0, x1), e in zip(ranuras, etapas):
        abierta = e["capa"] == capa
        rotulos.append({"capa": e["capa"], "corta": e["corta"],
                        "xm": (x0 + x1) / 2})
        alto = float(e["lado"]) or _ALTO_FC
        barras.append({
            "capa": e["capa"], "forma": e["forma"], "canales": e["canales"],
            "x0": x0, "x1": x1,
            "y0": -_PLACA / 2 if abierta else -alto / 2,
            "y1": _PLACA / 2 if abierta else alto / 2,
            "estado": "abierta" if abierta else "cerrada",
        })

        if not abierta or not valores:
            continue

        # La rejilla, dentro de la placa.
        ix0, ix1 = x0 + _MARGEN_X, x1 - _MARGEN_X
        iy0, iy1 = -_PLACA / 2 + _MARGEN_Y, _PLACA / 2 - _MARGEN_Y
        cols = _columnas(len(valores), ix1 - ix0, iy1 - iy0)
        filas = math.ceil(len(valores) / cols)
        ancho_c = (ix1 - ix0) / cols
        alto_c = (iy1 - iy0) / filas
        for i, v in enumerate(valores):
            c, f = i % cols, i // cols
            celdas.append({
                "canal": i,
                "valor": round(float(v), 4),
                # Un pelo de aire entre celdas, para que se vea la rejilla.
                "x0": ix0 + c * ancho_c, "x1": ix0 + (c + 1) * ancho_c - 0.6,
                "y1": iy1 - f * alto_c, "y0": iy1 - (f + 1) * alto_c + 0.35,
                "marca": ("elegido" if i == canal else
                          "destacado" if i in destacados else "normal"),
            })

    eje_x = {"field": "x0", "type": "quantitative", "axis": None,
             "scale": {"domain": [0, _ANCHO], "nice": False}}
    eje_y = {"field": "y0", "type": "quantitative", "axis": None,
             "scale": {"domain": [_Y_MIN, _Y_MAX], "nice": False}}

    capas_grafico = [
        # 1. Las etapas. Llevan el parametro de seleccion del primer nivel.
        {
            "data": {"values": barras},
            "mark": {"type": "rect", "cornerRadius": 4, "cursor": "pointer"},
            "params": [{
                "name": PARAM_ETAPA,
                "select": {"type": "point", "fields": ["capa"], "on": "click"},
            }],
            "encoding": {
                "x": eje_x, "x2": {"field": "x1"},
                "y": eje_y, "y2": {"field": "y1"},
                # El numero de canales da la intensidad. El rango va
                # explicito y no por esquema: el extremo bajo de 'teals' es
                # casi blanco y layer1 desaparecia contra el fondo.
                "color": {"field": "canales", "type": "quantitative",
                          "scale": {"range": ["#8ed0bf", "#0f6f5c"]},
                          "legend": None},
                "opacity": {
                    "condition": {"test": "datum.estado === 'abierta'",
                                  "value": 0.18},
                    "value": 0.75,
                },
                "tooltip": [{"field": "capa", "title": "capa"},
                            {"field": "forma", "title": "salida"}],
            },
        },
        # 2. Los canales de la etapa abierta, dentro de su placa.
        {
            "data": {"values": celdas},
            "mark": {"type": "rect", "cornerRadius": 1, "cursor": "pointer"},
            "params": [{
                "name": PARAM_CANAL,
                "select": {"type": "point", "fields": ["canal"],
                           "on": "click"},
            }],
            "encoding": {
                "x": eje_x, "x2": {"field": "x1"},
                "y": eje_y, "y2": {"field": "y1"},
                # El mismo mapa de color que el mapa de calor de la foto: en
                # las dos vistas, claro significa lo mismo.
                "color": {"field": "valor", "type": "quantitative",
                          "scale": {"scheme": "inferno"},
                          "legend": {"title": "activacion",
                                     "gradientLength": 110}},
                "stroke": {
                    "condition": [
                        {"test": "datum.marca === 'elegido'",
                         "value": "#3ebd9d"},
                        # Oscuro, no claro: las mas activas son por
                        # definicion las mas claras del mapa inferno, y un
                        # borde claro sobre amarillo no se ve.
                        {"test": "datum.marca === 'destacado'",
                         "value": "#1b1c19"},
                    ],
                    "value": None,
                },
                "strokeWidth": {
                    "condition": [
                        {"test": "datum.marca === 'elegido'", "value": 2},
                        {"test": "datum.marca === 'destacado'", "value": 1},
                    ],
                    "value": 0,
                },
                "tooltip": [{"field": "canal", "title": "canal"},
                            {"field": "valor", "title": "activacion media"}],
            },
        },
        # 3. Los rotulos, debajo de cada etapa.
        {
            "data": {"values": rotulos},
            "mark": {"type": "text", "fontSize": 12, "baseline": "middle"},
            "encoding": {
                "x": {"field": "xm", "type": "quantitative", "axis": None,
                      "scale": {"domain": [0, _ANCHO], "nice": False}},
                "y": {"datum": _Y_ROTULO, "type": "quantitative",
                      "axis": None,
                      "scale": {"domain": [_Y_MIN, _Y_MAX], "nice": False}},
                "text": {"field": "corta"},
                "color": {
                    "condition": {"test": f"datum.capa === '{capa}'",
                                  "value": "#0f6f5c"},
                    "value": "#9ba195",
                },
            },
        },
    ]

    return {
        "layer": capas_grafico,
        "height": _ALTO_PX,
        "config": {"view": {"stroke": None}},
        "resolve": {"scale": {"color": "independent"}},
    }
