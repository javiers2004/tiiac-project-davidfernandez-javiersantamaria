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
def _analizar(foto_bytes: bytes, capa: str, top_k: int):
    """Prediccion y ranking de neuronas. Se cachea por foto y capa.

    Recibe los bytes y no el fichero subido porque st.cache_data necesita un
    argumento hasheable; el objeto de Streamlit no lo es.
    """
    foto = Image.open(io.BytesIO(foto_bytes)).convert("RGB")
    m = _modelo()
    tensor = torch.from_numpy(np.array(foto)).permute(2, 0, 1).float().div(255)[None]
    predicciones = predecir(tensor, top_k=3, modelo=m)
    neuronas = listar_neuronas(tensor, capas=capa, top_k=top_k, modelo=m)
    return predicciones, neuronas, tensor


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

capa = st.segmented_control("Capa", CAPAS, default="layer3.1.conv2",
                            help="Todas son convoluciones pre-ReLU salvo fc. "
                                 "Las salidas de bloque tienen la mitad de las "
                                 "unidades muertas al arrancar.")
if capa is None:
    capa = "layer3.1.conv2"

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
    predicciones, neuronas, tensor = _analizar(foto_bytes, capa, 8)

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

    etiquetas = [f"{a.canal} ({a.valor:+.2f})" for a in neuronas]
    elegida_txt = st.radio("Elige una neurona", etiquetas, index=0,
                           label_visibility="collapsed")
    elegida: Activacion = neuronas[etiquetas.index(elegida_txt)]

    st.caption("El orden solo es comparable dentro de una capa: layer1 activa "
               "de media cuatro veces mas que layer3.")

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
