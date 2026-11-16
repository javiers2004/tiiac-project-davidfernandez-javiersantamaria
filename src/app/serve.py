"""Capa de servicio: el aplicativo que va encima de la replicacion.

    streamlit run src/app/serve.py

Carga un checkpoint y responde. NO entrena, NO conoce el dataset de
entrenamiento, NO importa nada de src/training/. Solo habla con la fachada
src/interpretability/api.py, que es el contrato.

El recorrido es el del articulo, de izquierda a derecha:

    una foto  ->  que especie ve el modelo        predecir()
              ->  que neuronas se encienden       listar_neuronas()
              ->  DONDE se encienden en la foto   mapa_activacion()
              ->  que patron busca esa neurona    generar_imagen()

El cuarto paso tarda entre 14 segundos y 3 minutos, asi que va detras de un
formulario y con barra de progreso: sin eso la interfaz se queda muda y
cualquier movimiento de un deslizador relanzaria la optimizacion entera.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.interpretability.api import (  # noqa: E402
    Activacion,
    generar_deepdream,
    cargar_modelo,
    generar_imagen,
    listar_neuronas,
    mapa_activacion,
    predecir,
    superponer_mapa,
)
from src.interpretability.feature_viz import UnidadMuerta, cargar_config  # noqa: E402
from src.interpretability.objetivos import inspeccionar_capa  # noqa: E402
from src.app.graficos import (  # noqa: E402
    PARAM_CANAL, PARAM_ETAPA, espec_red)

# Convoluciones pre-ReLU. Las salidas de bloque (layer3 a secas) son post-ReLU
# y arrancan en cero en mas de la mitad de las posiciones de layer4: elegir una
# de esas devuelve ruido sin avisar. Ver UnidadMuerta en feature_viz.py.
CAPAS = ["layer1.1.conv2", "layer2.1.conv2", "layer3.1.conv2",
         "layer4.1.conv2", "fc"]

st.set_page_config(page_title="Visualizacion de caracteristicas",
                   page_icon=":material/psychology:", layout="wide")


@st.cache_resource
def _modelo():
    """El checkpoint se lee una vez por sesion: son 94 MB."""
    return cargar_modelo()


@st.cache_data(show_spinner=False, max_entries=16)
def _analizar(foto_bytes: bytes, capa: str):
    """Prediccion y activacion de TODOS los canales. Se cachea por foto y capa.

    Recibe los bytes y no el fichero subido porque st.cache_data necesita un
    argumento hasheable; el objeto de Streamlit no lo es.

    Devuelve el ranking (de mayor a menor, que es lo que hace falta para
    destacar los mas activos) y los mismos valores en orden de indice de
    canal, que es lo que necesita la rejilla pulsable.
    """
    foto = Image.open(io.BytesIO(foto_bytes)).convert("RGB")
    m = _modelo()
    tensor = torch.from_numpy(np.array(foto)).permute(2, 0, 1).float().div(255)[None]
    predicciones = predecir(tensor, top_k=3, modelo=m)
    # Un top_k por encima del numero de canales: listar_neuronas lo recorta.
    ranking = listar_neuronas(tensor, capas=capa, top_k=10_000, modelo=m)
    valores = [0.0] * len(ranking)
    for a in ranking:
        valores[a.canal] = a.valor
    return predicciones, ranking, valores, tensor


@st.cache_data(show_spinner=False)
def _formas(tamano: int):
    """Forma de salida de cada capa seleccionable, para dibujar la red."""
    m = _modelo()
    salida = []
    for c in CAPAS:
        f = inspeccionar_capa(m, c, tamano)
        salida.append({
            "capa": c,
            "corta": c.split(".")[0],
            "canales": f.canales,
            "lado": f.alto if f.es_espacial else 0,
            "forma": (f"{f.canales} x {f.alto} x {f.ancho}" if f.es_espacial
                      else f"{f.canales} clases"),
        })
    return salida


def _seleccion(evento, parametro: str, campo: str):
    """Saca el valor pulsado de un grafico, o None si no se pulso nada.

    La forma del evento depende de la version, asi que en vez de confiar en
    una se comprueban las dos que devuelve Streamlit: una lista de filas, o
    un diccionario de columnas.
    """
    try:
        sel = evento.selection[parametro]
    except (AttributeError, KeyError, TypeError):
        return None
    if isinstance(sel, dict):
        valores = sel.get(campo) or []
        return valores[0] if valores else None
    if isinstance(sel, list) and sel:
        return sel[0].get(campo) if isinstance(sel[0], dict) else None
    return None


st.title("Visualizacion de caracteristicas")
st.caption("Olah, Mordvintsev & Schubert (2017) sobre un ResNet-18 "
           "preentrenado en Pl@ntNet-300K. Los pesos van congelados: lo que "
           "se optimiza es la imagen.")

cfg = cargar_config(RAIZ / "configs" / "feature_viz.yaml")

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.subheader("Controles del articulo")
    st.caption("Los nueve mandos de sus figuras interactivas.")

    pasos = st.slider("Pasos de optimizacion", 32, 1024,
                          int(cfg["optimizacion"]["pasos"]), step=32,
                          help="El articulo usa 2560. Cada paso son unos 70 ms en CPU.")
    lr = st.select_slider("Learning rate", [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
                              value=float(cfg["optimizacion"]["learning_rate"]))

    st.markdown("**Penalizadores de frecuencia**")
    l1 = st.slider("L1", 0.0, 8.0, 0.0, step=0.25)
    tv = st.slider("Variacion total", 0.0, 20.0, 0.0, step=0.5,
                   help="El mas efectivo de los tres: baja la aspereza de 0.16 a 0.04.")
    blur = st.slider("Desenfoque", 0.0, 200.0, 0.0, step=5.0)

    st.markdown("**Robustez a transformaciones**")
    robustez = st.toggle("Activada", value=True,
                         help="Apagarla sube la activacion y empeora la imagen. "
                              "Es el argumento central del articulo.")
    jitter = st.slider("Jitter (px)", 0, 32, int(cfg["robustez"]["jitter"]))
    rot = st.slider("Rotacion maxima (grados)", 0, 40, 5)
    escala = st.slider("Amplitud de escala", 0.0, 0.4, 0.05, step=0.025)

    st.markdown("**Preconditioning**")
    precond = st.toggle("Espacio de Fourier decorrelado", value=True,
                        help="Apagarlo optimiza los pixeles directamente "
                             "y converge a ruido adversarial.")

    st.divider()
    st.subheader("Seguimiento")
    registrar_wandb = st.toggle(
        "Registrar en W&B", value=False,
        help="Apagado por defecto: encendido, cada pulsacion de Generar crea "
             "un run. Enciendelo para enseñar la integracion en la demo.")
    st.link_button("Ver los runs en W&B",
                   "https://wandb.ai/javier-santamaria-p-deusto/tiiac-2026",
                   icon=":material/monitoring:", width="stretch")

# -------------------------------------------------------------------- la foto
muestras = sorted((RAIZ / "data" / "muestras").glob("*.jpg"))

with st.container(horizontal=True, vertical_alignment="bottom"):
    subida = st.file_uploader("Sube una foto de una planta", type=["jpg", "jpeg", "png"])
    usar_muestra = st.selectbox(
        "O elige una del conjunto de test",
        ["(ninguna)"] + [p.name.split("_")[0] for p in muestras],
        help="El nombre es el id de especie de Pl@ntNet, que hace de verdad conocida.")

if subida is not None:
    foto_bytes = subida.getvalue()
elif usar_muestra != "(ninguna)":
    foto_bytes = next(p for p in muestras
                      if p.name.startswith(usar_muestra)).read_bytes()
else:
    st.info("Sube una foto o elige una muestra para empezar.",
            icon=":material/photo_camera:")
    st.stop()

st.subheader("La red, por etapas")
st.caption("Pulsa una etapa para inspeccionarla. La altura es el lado de su "
           "mapa de activacion y el color, cuantos canales tiene: la "
           "resolucion cae de 56 a 7 mientras los canales pasan de 64 a 512. "
           "Todas son convoluciones pre-ReLU salvo fc.")

st.session_state.setdefault("capa_sel", "layer3.1.conv2")
capa = st.session_state["capa_sel"]

# El grafico se pinta DENTRO de este hueco, pero mas abajo en el script: la
# etapa abierta lleva su mapa de activacion dentro, y el mapa no se conoce
# hasta haber elegido el canal. Reservar el sitio aqui mantiene el orden de
# la pagina sin obligar a calcularlo todo antes de tiempo.
hueco_red = st.container()

# Los objetivos del articulo. 'capa' no elige unidad: maximiza la capa entera.
# 'logit' y 'probabilidad' solo tienen sentido en fc, donde el canal ES la clase.
OBJETIVOS = {"canal": "Channel", "neurona": "Neuron", "capa": "Layer (DeepDream)",
             "logit": "Class Logits", "probabilidad": "Class Probability"}
disponibles = ([k for k in OBJETIVOS if k in ("logit", "probabilidad", "capa")]
               if capa == "fc" else ["canal", "neurona", "capa"])
tipo = st.segmented_control(
    "Objetivo", disponibles, default=disponibles[0],
    format_func=lambda k: OBJETIVOS[k],
    help="Los objetivos de la seccion 'Optimization Objectives' del articulo.")
if tipo is None:
    tipo = disponibles[0]

with st.spinner("Pasando la foto por el modelo..."):
    predicciones, ranking, valores, tensor = _analizar(foto_bytes, capa)

# ------------------------------------------------------- prediccion y neuronas
izq, der = st.columns([1, 1])

with izq:
    st.subheader("Que ve el modelo")
    st.image(foto_bytes, width="stretch", alt="Foto de la planta a analizar")
    primera, *resto = predicciones
    st.metric(primera.nombre or f"clase {primera.clase}",
              f"{primera.probabilidad:.1%}", border=True)
    for p in resto:
        st.caption(f"{p.nombre or p.clase} — {p.probabilidad:.1%}")

with der:
    st.subheader(f"Neuronas de {capa}")
    forma = inspeccionar_capa(_modelo(), capa, int(cfg["imagen"]["tamano"]))
    st.caption(f"{forma.canales} canales"
               + (f", mapa de {forma.alto}x{forma.ancho}" if forma.es_espacial
                  else ", sin extension espacial"))

    top = ranking[:8]
    destacados = {a.canal for a in top}
    st.session_state.setdefault("canal_sel", top[0].canal)
    # Al cambiar de capa el canal elegido puede no existir en la nueva.
    if not 0 <= st.session_state["canal_sel"] < len(valores):
        st.session_state["canal_sel"] = top[0].canal
    canal = st.session_state["canal_sel"]

    if not forma.es_espacial:
        # fc tiene 1081 canales, que son las clases: una rejilla de 1081
        # celdas dentro de la etapa no se puede pulsar. Ahi la lista sigue
        # siendo lo mejor.
        etiquetas = [f"{a.canal} ({a.valor:+.2f})" for a in top]
        elegida_txt = st.radio("Elige una clase", etiquetas, index=0,
                               label_visibility="collapsed")
        canal = top[etiquetas.index(elegida_txt)].canal
        st.session_state["canal_sel"] = canal
    else:
        # La rejilla va dentro de la etapa abierta, arriba. Aqui solo las
        # cifras del ranking, que en la rejilla solo salen al pasar por
        # encima.
        st.caption("Las ocho mas activas, que en la rejilla de arriba van "
                   "con borde:")
        for a in top:
            marca = "**" if a.canal == canal else ""
            st.caption(f"{marca}canal {a.canal} — {a.valor:+.3f}{marca}")

    elegida = Activacion(capa, int(canal), float(valores[int(canal)]))
    st.caption(f"Elegido el canal **{elegida.canal}**, activacion "
               f"{elegida.valor:+.3f}. El orden solo es comparable dentro de "
               "una capa: layer1 activa de media cuatro veces mas que layer3.")

# --------------------------------------------- la red, ya con sus canales
# Se dibuja aqui, en el hueco reservado arriba, porque la etapa abierta
# lleva dentro la rejilla de canales y esa no se conoce hasta haber hecho
# la pasada por el modelo.
#
# La clave lleva la capa a proposito: con una clave fija, al volver a
# ejecutarse el script Streamlit reutiliza el elemento y el dibujo se queda
# en la etapa anterior aunque el resto de la pagina ya haya cambiado.
_ev_red = hueco_red.vega_lite_chart(
    espec_red(_formas(int(cfg["imagen"]["tamano"])), capa,
              valores if forma.es_espacial else None,
              int(canal), destacados),
    on_select="rerun", key=f"grafico_red_{capa}")

_pulsada = _seleccion(_ev_red, PARAM_ETAPA, "capa")
if _pulsada and _pulsada != capa:
    st.session_state["capa_sel"] = _pulsada
    # Otra capa tiene otros canales: el que estuviera elegido ya no vale.
    st.session_state.pop("canal_sel", None)
    st.rerun()

_pulsado = _seleccion(_ev_red, PARAM_CANAL, "canal")
if _pulsado is not None and int(_pulsado) != int(canal):
    st.session_state["canal_sel"] = int(_pulsado)
    st.rerun()

# -------------------------------------------------------------- mapa y patron
st.divider()
mapa_col, patron_col = st.columns(2)

with mapa_col:
    st.subheader("Donde se enciende")
    if forma.es_espacial:
        mapa = mapa_activacion(tensor, elegida, modelo=_modelo())
        sobre = superponer_mapa(tensor, mapa)
        st.image(sobre.permute(1, 2, 0).numpy(), width="stretch",
                 alt=f"Mapa de calor del canal {elegida.canal} sobre la foto")
        st.caption("Claro = la neurona se dispara ahi. Es la rejilla del canal "
                   "antes del promedio que hace el ranking.")
    else:
        st.info("La capa fc no tiene extension espacial, asi que no hay mapa.",
                icon=":material/info:")

with patron_col:
    st.subheader("Que patron busca")
    hueco = st.container()
    lanzar = st.button("Generar", icon=":material/auto_awesome:",
                       type="primary", width="stretch")
    st.caption(f"{pasos} pasos, unos {pasos * 0.07:.0f} s en CPU.")

clave = (capa, tipo, elegida.canal, pasos, lr, l1, tv, blur, robustez, jitter, rot,
         escala, precond)
st.session_state.setdefault("patron", None)
st.session_state.setdefault("clave_patron", None)

if lanzar:
    c = {**cfg,
         "optimizacion": {**cfg["optimizacion"], "pasos": pasos, "learning_rate": lr},
         "regularizacion": {**cfg["regularizacion"], "l1": l1, "tv": tv, "blur": blur},
         "imagen": {**cfg["imagen"],
                    "parametrizacion": "fourier" if precond else "pixel"},
         "robustez": {**cfg["robustez"], "activa": robustez, "jitter": jitter,
                      "jitter_final": max(0, jitter // 2),
                      "rotaciones": list(range(-rot, rot + 1)) or [0],
                      "escalas": [1.0] if escala == 0 else
                                 [round(1 + escala * k / 2, 4) for k in range(-2, 3)]},
         }

    barra = hueco.progress(0.0, text="Optimizando...")
    vista = hueco.empty()

    def progreso(paso, total, act, imagen):
        # Cada diez pasos: repintar en cada uno cuesta mas que el propio paso.
        if paso % max(1, total // 10) == 0 or paso == total - 1:
            barra.progress((paso + 1) / total,
                           text=f"Paso {paso + 1} de {total} &middot; activacion {act:+.2f}")
            vista.image(imagen[0].permute(1, 2, 0).numpy(), width="stretch",
                        alt="Vista previa de la optimizacion en curso")

    try:
        if tipo == "capa":
            # DeepDream no elige unidad: maximiza el cuadrado de la capa entera.
            imagen = generar_deepdream(capa, pasos=pasos, modelo=_modelo(), cfg=c,
                                       al_progresar=progreso,
                                       registrar=registrar_wandb)
        else:
            imagen = generar_imagen(elegida, tipo=tipo, pasos=pasos,
                                    modelo=_modelo(), cfg=c,
                                    al_progresar=progreso,
                                    registrar=registrar_wandb)
        st.session_state["patron"] = imagen[0].permute(1, 2, 0).numpy()
        st.session_state["clave_patron"] = clave
        if registrar_wandb:
            st.toast("Run registrado en W&B", icon=":material/monitoring:")
    except UnidadMuerta as e:
        st.session_state["patron"] = None
        hueco.error(str(e), icon=":material/warning:")
    finally:
        barra.empty()
        vista.empty()

if st.session_state["patron"] is not None:
    hueco.image(st.session_state["patron"], width="stretch",
                alt=f"Patron optimizado para el canal {elegida.canal}")
    if st.session_state["clave_patron"] != clave:
        hueco.caption(":orange[Los parametros han cambiado desde que se genero. "
                      "Vuelve a pulsar Generar.]")
elif not lanzar:
    hueco.info("Pulsa Generar para optimizar una imagen desde ruido.",
               icon=":material/image:")
