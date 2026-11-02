"""Visualizacion de caracteristicas por maximizacion de activacion.

Elige una neurona, parte de ruido y optimiza LA IMAGEN DE ENTRADA (no los
pesos) para maximizar la activacion de esa neurona. El resultado es el patron
que esa neurona busca.

    python -m src.interpretability.feature_viz --neurona 0
    python -m src.interpretability.feature_viz --capa layer3.1.conv2 --neurona 42
    python -m src.interpretability.feature_viz --tipo capa --capa layer3
    python -m src.interpretability.feature_viz --tipo logit --capa fc --canal 7
    python -m src.interpretability.feature_viz --tipo probabilidad --capa fc --canal 7
    python -m src.interpretability.feature_viz --tipo direccion --capa layer4.1.conv2 --direccion 3
    python -m src.interpretability.feature_viz --tipo suma --capa layer4 --canales 12 40
    python -m src.interpretability.feature_viz --tipo interpolacion --capa layer4 --canales 12 40 --lote 6
    python -m src.interpretability.feature_viz --capa layer4 --canal 443 --lote 4 --diversidad

Tres detalles sin los cuales esto NO funciona, y que son casi toda la calidad
del resultado:

  1. Parametrizacion en frecuencia (FFT) con decorrelacion de color. Optimizar
     los pixeles directamente converge a ruido adversarial de alta frecuencia:
     maximiza la activacion pero no se parece a nada. Al optimizar el espectro
     con las frecuencias altas atenuadas, el gradiente se reparte entre
     escalas y salen patrones. Es la tecnica de Olah et al., "Feature
     Visualization" (Distill, 2017).
  2. Robustez a transformaciones. En cada paso se aplican jitter, escala y
     rotacion aleatorios. Obliga a la imagen a activar la neurona desde
     cualquier encuadre, y eso elimina el ruido que solo funciona en una
     posicion exacta.
  3. Para clases se maximiza el LOGIT, no la softmax. Maximizar la
     probabilidad premia hundir las otras 1080 clases, que es mas facil que
     representar la propia. El objetivo 'probabilidad' existe para replicar
     esa comparacion del articulo, no para usarlo por defecto.

Los pesos van congelados: el unico tensor con gradiente es la imagen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from torchvision.models import resnet18, resnet50, resnet152
from torchvision.utils import save_image

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.interpretability.objetivos import (        # noqa: E402
    OBJETIVOS,
    Interpolacion,
    construir_objetivo,
    inspeccionar_capa,
    resolver_modulo,
)
from src.utils.seed import fijar_seed               # noqa: E402
from src.utils.tracking import (abrir_run, registrar,  # noqa: E402
                                registrar_imagen)
from src.utils.utils import load_model              # noqa: E402

ARQUITECTURAS = {"resnet18": resnet18, "resnet50": resnet50, "resnet152": resnet152}

# Matriz de decorrelacion de color de Olah et al. Los canales RGB de las
# imagenes naturales estan muy correlacionados; optimizar en la base
# decorrelacionada evita que la imagen derive hacia un tinte uniforme.
_SVD_COLOR = np.asarray([[0.26, 0.09, 0.02],
                         [0.27, 0.00, -0.05],
                         [0.27, -0.09, 0.03]], dtype=np.float32)
_COLOR_NORM = _SVD_COLOR / np.max(np.linalg.norm(_SVD_COLOR, axis=0))


class ImagenFourier(nn.Module):
    """Imagen parametrizada por su espectro, con las altas frecuencias atenuadas.

    El tensor que se optimiza no son los pixeles sino los coeficientes de
    Fourier. Cada coeficiente se escala por 1/f**decay antes de la transformada
    inversa, asi que un paso de gradiente del mismo tamano mueve mucho mas las
    frecuencias bajas que las altas. Eso es lo que evita el ruido adversarial.

    Con lote > 1 se optimizan varias imagenes a la vez, cada una con su propio
    ruido de partida. Es lo que necesita la interpolacion entre neuronas, donde
    cada imagen de la fila tiene un peso distinto.

    `compartida` anade la parte COMPARTIDA de la interpolacion del articulo
    ("a combination of separate and shared image parameterizations"): tensores
    de baja resolucion comunes a todo el lote que se suman a la parte propia
    de cada imagen. Cada uno es (imagenes, divisor): un tensor de
    alto/divisor x ancho/divisor con `imagenes` fotogramas clave, reescalado
    al tamano real y, a lo largo del lote, interpolado entre fotogramas. Con
    imagenes=1 es identico en toda la fila; con 2 va cambiando suavemente de
    un extremo al otro. Facilita que los objetos queden en el mismo sitio sin
    impedir que cada imagen sea distinta. Solo se usa con lote > 1.

    Referencia: lucid.optvis.param.lowres_tensor y el interpolate_param_f del
    notebook neuron_interaction.ipynb de lucid.
    """

    def __init__(self, alto: int, ancho: int, decay: float = 1.0,
                 sd: float = 0.01, generador: torch.Generator | None = None,
                 lote: int = 1, compartida: list | None = None):
        super().__init__()
        self.alto, self.ancho, self.lote = alto, ancho, lote
        # Con lote=1 el ruido es exactamente el de antes de admitir lotes.
        ruido = torch.randn(lote, 3, alto, ancho // 2 + 1, 2, generator=generador)
        self.espectro = nn.Parameter(ruido * sd)
        self.register_buffer("escala", self._escala_frecuencias(alto, ancho, decay))

        # Se crea DESPUES del espectro para no alterar su ruido inicial.
        compartida = compartida if lote > 1 else []
        self.compartida = nn.ParameterList(
            nn.Parameter(sd * torch.randn(1, 3, int(n), alto // int(div),
                                          ancho // int(div), generator=generador))
            for n, div in (compartida or [])
        )

    @staticmethod
    def _escala_frecuencias(alto: int, ancho: int, decay: float) -> torch.Tensor:
        """Devuelve el factor 1/f**decay para cada frecuencia del espectro."""
        fy = np.fft.fftfreq(alto)[:, None]
        fx = np.fft.fftfreq(ancho)[: ancho // 2 + 1][None, :]
        frecuencias = np.sqrt(fx ** 2 + fy ** 2)
        # El suelo evita dividir por cero en la componente continua (f=0).
        suelo = 1.0 / max(alto, ancho)
        escala = 1.0 / np.maximum(frecuencias, suelo) ** decay
        escala = escala * np.sqrt(alto * ancho)
        return torch.tensor(escala, dtype=torch.float32)

    def forward(self) -> torch.Tensor:
        """Reconstruye las imagenes RGB (lote, 3, H, W) en [0, 1]."""
        espectro = torch.view_as_complex(self.espectro) * self.escala
        imagen = torch.fft.irfft2(espectro, s=(self.alto, self.ancho))
        imagen = imagen / 4.0  # constante empirica de Olah et al.
        for baja in self.compartida:
            # (1, 3, fotogramas, h, w) -> (1, 3, lote, H, W) -> (lote, 3, H, W)
            subida = F.interpolate(baja, size=(self.lote, self.alto, self.ancho),
                                   mode="trilinear", align_corners=False)
            imagen = imagen + subida[0].permute(1, 0, 2, 3)

        # Decorrelacion de color: (B,3,H,W) -> (N,3) -> matmul -> vuelta.
        color = torch.as_tensor(_COLOR_NORM, device=imagen.device)
        plano = imagen.permute(0, 2, 3, 1).reshape(-1, 3) @ color.T
        imagen = plano.view(self.lote, self.alto, self.ancho, 3).permute(0, 3, 1, 2)

        # La sigmoide mantiene la imagen en [0,1] sin matar el gradiente, que
        # es lo que haria un clamp en cuanto la imagen tocase los extremos.
        return torch.sigmoid(imagen)


class ImagenPixel(nn.Module):
    """Imagen parametrizada por sus pixeles, sin preconditioning.

    Es el interruptor "Decorrelated Space" del articulo en posicion apagada:
    descenso de gradiente en el espacio de pixeles, sin base de Fourier y sin
    decorrelacion de color. Existe para poder REPRODUCIR la comparacion del
    articulo, no para usarla: converge a ruido adversarial de alta frecuencia
    que maximiza la activacion sin parecerse a nada.

    Equivale a param.pixel_image de lucid, con la misma sd por defecto.
    """

    def __init__(self, alto: int, ancho: int, sd: float = 0.01,
                 generador: torch.Generator | None = None, lote: int = 1):
        super().__init__()
        self.alto, self.ancho, self.lote = alto, ancho, lote
        ruido = torch.randn(lote, 3, alto, ancho, generator=generador)
        self.pixeles = nn.Parameter(ruido * sd)

    def forward(self) -> torch.Tensor:
        """Devuelve las imagenes RGB (lote, 3, H, W) en [0, 1]."""
        # Misma sigmoide que la version de Fourier, para que la unica
        # diferencia entre ambas sea la parametrizacion.
        return torch.sigmoid(self.pixeles)


def construir_parametrizacion(icfg: dict, generador: torch.Generator
                              ) -> nn.Module:
    """Crea la parametrizacion de la imagen segun icfg['parametrizacion'].

    'fourier' es el preconditioning del articulo y el valor por defecto;
    'pixel' lo desactiva.
    """
    tam, lote = int(icfg["tamano"]), int(icfg.get("lote", 1))
    sd = float(icfg["sd_inicial"])
    cual = str(icfg.get("parametrizacion", "fourier"))

    if cual == "pixel":
        return ImagenPixel(tam, tam, sd=sd, generador=generador, lote=lote)
    if cual != "fourier":
        raise ValueError(f"parametrizacion desconocida: '{cual}'. "
                         f"Opciones: fourier, pixel")
    return ImagenFourier(tam, tam, decay=float(icfg["decay_power"]), sd=sd,
                         generador=generador, lote=lote,
                         compartida=icfg.get("compartida"))


def aplicar_robustez(imagen: torch.Tensor, rcfg: dict,
                     generador: torch.Generator) -> torch.Tensor:
    """Aplica jitter, escala y rotacion aleatorios a la imagen.

    Args:
        imagen: tensor (N, 3, H, W) en [0, 1]. Todas las imagenes del lote
            reciben la MISMA transformacion, como en lucid: asi las de una
            interpolacion siguen alineadas entre si.
        rcfg: subconfiguracion 'robustez'.
        generador: generador propio, para no tocar el estado global de torch.

    Returns:
        La imagen transformada, del mismo tamano espacial que la original.
    """
    alto, ancho = imagen.shape[-2:]

    pad = int(rcfg["pad"])
    if pad:
        imagen = F.pad(imagen, (pad,) * 4, mode="reflect")

    imagen = _desplazar(imagen, int(rcfg["jitter"]), generador)

    escalas = rcfg["escalas"]
    if escalas:
        e = float(escalas[_aleatorio(len(escalas), generador)])
        nuevo = (max(1, int(imagen.shape[-2] * e)), max(1, int(imagen.shape[-1] * e)))
        imagen = F.interpolate(imagen, size=nuevo, mode="bilinear", align_corners=False)

    rotaciones = rcfg["rotaciones"]
    if rotaciones:
        angulo = float(rotaciones[_aleatorio(len(rotaciones), generador)])
        imagen = TF.rotate(imagen, angulo,
                           interpolation=TF.InterpolationMode.BILINEAR)

    imagen = _desplazar(imagen, int(rcfg["jitter_final"]), generador)
    return TF.center_crop(imagen, [alto, ancho])


def _aleatorio(n: int, generador: torch.Generator) -> int:
    return int(torch.randint(0, n, (1,), generator=generador).item())


def _desplazar(x: torch.Tensor, maximo: int, generador: torch.Generator) -> torch.Tensor:
    if maximo <= 0:
        return x
    dy = _aleatorio(2 * maximo + 1, generador) - maximo
    dx = _aleatorio(2 * maximo + 1, generador) - maximo
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def construir_modelo_congelado(mcfg: dict) -> nn.Module:
    """Carga el checkpoint y deja el modelo en eval con los pesos congelados."""
    arquitectura = mcfg["arquitectura"]
    if arquitectura not in ARQUITECTURAS:
        raise ValueError(f"arquitectura desconocida: {arquitectura}")

    ruta = RAIZ / mcfg["checkpoint"]
    if not ruta.is_file():
        raise FileNotFoundError(f"no existe el checkpoint {mcfg['checkpoint']}")

    modelo = ARQUITECTURAS[arquitectura](num_classes=int(mcfg["n_clases"]))
    load_model(modelo, filename=str(ruta), use_gpu=torch.cuda.is_available())

    # Congelar es el punto del ejercicio: el gradiente tiene que llegar a la
    # imagen, no a los pesos.
    modelo.eval()
    modelo.requires_grad_(False)
    return modelo


def maximizar_activacion(cfg: dict, run: Any | None = None,
                         modelo: nn.Module | None = None,
                         objetivo: Any | None = None,
                         al_progresar: Callable[[int, int, float,
                                                 torch.Tensor], Any] | None = None
                         ) -> tuple[torch.Tensor, list[float]]:
    """Optimiza una imagen de ruido para maximizar la activacion de una neurona.

    Args:
        cfg: configuracion completa, tal cual sale del YAML.
        run: run de W&B abierto, o None.
        modelo: modelo ya congelado. Si es None se carga el del checkpoint.
            Se inyecta desde los tests, que no tienen los .tar disponibles.
        al_progresar: se llama en cada paso con (paso, pasos, activacion,
            imagen), donde imagen es el estado actual (lote,3,H,W) en [0,1],
            desconectada del grafo. Si devuelve False la optimizacion para y
            se devuelve lo que haya hasta ese momento.

            Existe porque esto bloquea entre 14 segundos y 3 minutos sin dar
            senales, y una interfaz no puede quedarse muda tanto rato. Es un
            callback y no un generador a proposito: asi no cambia el `return`,
            y ningun llamador existente (la CLI, las seis funciones de api.py,
            los tests) tiene que tocarse.

    Returns:
        (imagen, historial) con las imagenes finales (lote,3,H,W) en [0,1] y
        la activacion alcanzada en cada paso. lote sale de imagen.lote en el
        YAML y es 1 si no esta.
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if modelo is None:
        modelo = construir_modelo_congelado(cfg["modelo"])
    modelo = modelo.to(dispositivo)

    ocfg, icfg, pcfg = cfg["objetivo"], cfg["imagen"], cfg["optimizacion"]
    capa = str(ocfg["capa"])
    # Si no llega uno ya construido (p.ej. una Suma desde api.py), sale del YAML.
    if objetivo is None:
        objetivo = construir_objetivo(ocfg)

    # Un generador propio: las semillas globales las fija src/utils/seed.py y
    # no se tocan desde aqui.
    generador = torch.Generator().manual_seed(int(cfg["experimento"]["seed"]))

    rcfg, gcfg = cfg["robustez"], cfg["regularizacion"]
    # Diversidad (seccion "Diversity"): apagada salvo que se pida. Compara las
    # imagenes del lote entre si, asi que con una sola ni se calcula.
    dcfg = gcfg.get("diversidad") or {}
    peso_div = float(dcfg.get("peso", 0.0)) if int(icfg.get("lote", 1)) > 1 else 0.0
    if peso_div:
        if isinstance(objetivo, Interpolacion):
            raise ValueError(
                "la diversidad separa las imagenes del lote y la interpolacion "
                "las quiere alineadas; no se combinan en la misma ejecucion")
        # La alineacion y la parte compartida juntan las imagenes del lote,
        # justo lo contrario. Se apagan SOLO en esta ejecucion; el YAML no se
        # toca y la interpolacion las sigue teniendo.
        print("diversidad activa: alineacion y parte compartida apagadas en "
              "esta ejecucion")
        icfg = {**icfg, "compartida": []}
    capa_div = str(dcfg.get("capa") or capa) if peso_div else None

    parametrizacion = construir_parametrizacion(icfg, generador).to(dispositivo)

    # Interruptor "Transformation Robustness" del articulo. Apagarlo no es lo
    # mismo que poner los valores a cero: aqui se salta la funcion entera.
    robustez_activa = bool(rcfg.get("activa", True))
    peso_tv, peso_l2 = float(gcfg["tv"]), float(gcfg["l2"])
    peso_l1 = float(gcfg.get("l1", 0.0))
    peso_blur = float(gcfg.get("blur", 0.0))
    acfg = gcfg.get("alineacion") or {}
    peso_alin = float(acfg.get("peso", 0.0))
    # La alineacion compara imagenes del lote entre si: con una sola seria
    # cero, asi que ni se calcula. Por eso puede tener peso en el YAML sin
    # afectar a los objetivos de una imagen.
    if parametrizacion.lote < 2 or peso_div:
        peso_alin = 0.0
    capas_alin = [str(c) for c in acfg["capas"]] if peso_alin else []

    # Los hooks capturan la salida de cada capa que hace falta: la del
    # objetivo y, si hay alineacion o diversidad, las suyas (pueden incluir
    # la misma).
    # Sirven igual para 'fc' (2D, N x clases) que para una conv (4D).
    capturado: dict[str, torch.Tensor] = {}
    asas = [
        resolver_modulo(modelo, c).register_forward_hook(
            lambda _m, _entrada, salida, c=c: capturado.__setitem__(c, salida))
        for c in dict.fromkeys(filter(None, [capa, *capas_alin, capa_div]))
    ]

    optimizador = torch.optim.Adam(parametrizacion.parameters(),
                                   lr=float(pcfg["learning_rate"]))
    ncfg = cfg["normalizacion"]
    media = torch.tensor(ncfg["media"], device=dispositivo).view(1, 3, 1, 1)
    desv = torch.tensor(ncfg["desviacion"], device=dispositivo).view(1, 3, 1, 1)

    pasos = int(pcfg["pasos"])
    # 0 lo desactiva. Subir imagenes cuesta tiempo y espacio en el servidor,
    # asi que por defecto van unas diez por run, no 256.
    cada_imagen = int(cfg.get("wandb", {}).get("cada_imagen", 0) or 0)
    historial: list[float] = []
    try:
        for paso in range(pasos):
            optimizador.zero_grad(set_to_none=True)

            imagen = parametrizacion()
            entrada = (aplicar_robustez(imagen, rcfg, generador)
                       if robustez_activa else imagen)
            modelo((entrada - media) / desv)

            # El objetivo decide la reduccion: que parte de la capa se toma
            # y como se resume en un escalar. Channel promedia un canal,
            # Neuron toma una posicion, Direction proyecta sobre un vector...
            # La lista entera, en la cabecera de objetivos.py.
            activacion = objetivo(capturado[capa])

            # Una unidad muerta (post-ReLU en cero) no da gradiente, y el
            # bucle devolveria el ruido inicial sin una sola queja. Mas de la
            # mitad de las posiciones centrales de layer4 estan asi al
            # arrancar, asi que conviene detectarlo en el primer paso. Se mira
            # el gradiente del OBJETIVO solo: las penalizaciones tambien mueven
            # la imagen y taparian una unidad muerta. Y en una Suma o una
            # Interpolacion, cada parte por separado, por el mismo motivo.
            if paso == 0:
                partes = (objetivo.componentes(capturado[capa])
                          if hasattr(objetivo, "componentes")
                          else [(str(objetivo), activacion)])
                for nombre, valor in partes:
                    _comprobar_gradiente(parametrizacion, valor, nombre, capa)

            perdida = -activacion
            metricas = {"viz/activacion": activacion.item()}
            # Las tres familias de "Three Families of Regularization": los
            # penalizadores de frecuencia (L1, variacion total, desenfoque)
            # aqui, y la robustez a transformaciones mas arriba.
            if peso_l1:
                perdida = perdida + peso_l1 * _norma_l1(imagen)
            if peso_tv:
                perdida = perdida + peso_tv * _variacion_total(imagen)
            if peso_blur:
                perdida = perdida + peso_blur * _penalizacion_desenfoque(imagen)
            if peso_l2:
                perdida = perdida + peso_l2 * imagen.pow(2).mean()
            if peso_alin:
                penal = sum(alineacion(capturado[c], float(acfg["decaimiento"]),
                                       int(acfg["alcance"])) for c in capas_alin)
                perdida = perdida + peso_alin * penal
                metricas["viz/alineacion"] = penal.item()
            if peso_div:
                similitud = diversidad(capturado[capa_div])
                perdida = perdida + peso_div * similitud
                metricas["viz/diversidad"] = similitud.item()

            perdida.backward()

            optimizador.step()

            # .item() rompe el grafo: sin esto se acumula memoria.
            valor = metricas["viz/activacion"]
            historial.append(valor)
            # La aspereza es lo que distingue un patron del ruido adversarial,
            # y es lo que tienen que bajar los penalizadores de frecuencia.
            # Sin ella, comparar dos runs obliga a mirar las imagenes a ojo.
            metricas["viz/aspereza"] = _variacion_total(imagen).item()
            registrar(run, {**metricas, "viz/perdida": perdida.item()}, paso)

            # La imagen cada cierto numero de pasos: es la figura
            # Step 1 / 32 / 128 / 256 del articulo, generada sola. Sin esto
            # el run solo tiene curvas y no se ve que salio.
            if cada_imagen and (paso % cada_imagen == 0 or paso == pasos - 1):
                with torch.no_grad():
                    registrar_imagen(run, "viz/imagen",
                                     parametrizacion().detach().cpu(), paso,
                                     f"paso {paso}")
            if paso % max(1, pasos // 10) == 0:
                print(f"paso {paso:4d}/{pasos}  activacion={valor:+.4f}")

            if al_progresar is not None:
                # La vista previa va detached y en cpu: si se le pasara el
                # tensor vivo, quien la guarde mantendria el grafo entero en
                # memoria paso tras paso.
                with torch.no_grad():
                    vista = parametrizacion().detach().cpu()
                if al_progresar(paso, pasos, valor, vista) is False:
                    break
    finally:
        for asa in asas:
            asa.remove()

    with torch.no_grad():
        final = parametrizacion().detach().cpu()
    return final, historial


class UnidadMuerta(RuntimeError):
    """La unidad elegida no propaga gradiente a la imagen.

    Se expone como excepcion propia para que la aplicacion pueda capturarla y
    mostrar un aviso en la interfaz en vez de un traceback.
    """


def _comprobar_gradiente(parametrizacion: nn.Module, activacion: torch.Tensor,
                         objetivo, capa: str) -> None:
    """Falla pronto si el objetivo no mueve la imagen.

    Se piden los gradientes de TODOS los parametros de la parametrizacion, no
    de uno concreto: el espectro de Fourier y los pixeles son tensores
    distintos, y la de Fourier ademas tiene la parte compartida del lote.
    """
    tensores = list(parametrizacion.parameters())
    grads = torch.autograd.grad(activacion, tensores,
                                retain_graph=True, allow_unused=True)
    if any(g is not None and torch.any(g != 0) for g in grads):
        return
    raise UnidadMuerta(
        f"{objetivo} en '{capa}' no propaga gradiente: la unidad esta muerta "
        f"(un ReLU la deja en cero) y la imagen no cambiaria.\n"
        f"Opciones: apuntar a la convolucion PRE-ReLU en vez de a la salida "
        f"del bloque (p.ej. '{capa}.1.conv2' en lugar de '{capa}'), elegir "
        f"otro canal, o mover (x, y) a otra posicion."
    )


def diversidad(act: torch.Tensor) -> torch.Tensor:
    """Cuanto se parecen entre si las imagenes del lote, por su "estilo".

    Es el "diversity term" de la seccion "Diversity" del articulo, sacado de
    la transferencia de estilo. Para cada imagen se calcula la matriz de Gram
    de los canales de la capa,

        G[i, j] = sum_{x, y} act[i, x, y] * act[j, x, y]

    que dice que canales se encienden JUNTOS pero no donde. Luego se suma la
    similitud coseno entre las Gram de cada par de imagenes distintas y se
    divide por el tamano del lote. Va de 0 (todas distintas) a lote - 1
    (todas iguales).

    Devuelve una penalizacion que se SUMA a la perdida: minimizarla separa
    las imagenes. El articulo escribe el termino con el signo cambiado porque
    alli se maximiza.

    Referencia: lucid.optvis.objectives.diversity.
    """
    if act.dim() != 4:
        raise ValueError(
            f"la diversidad compara matrices de Gram y necesita una capa con "
            f"extension espacial (N,C,H,W); esta da {tuple(act.shape)}. Con un "
            f"objetivo en fc, pon regularizacion.diversidad.capa (p.ej. layer4).")
    n = act.shape[0]
    plano = act.reshape(n, act.shape[1], -1)                  # (N, C, H*W)
    gram = (plano @ plano.transpose(1, 2)).reshape(n, -1)     # (N, C*C)
    gram = F.normalize(gram, dim=1, eps=1e-10)
    similitud = gram @ gram.T                                 # (N, N) cosenos
    return (similitud.sum() - similitud.diagonal().sum()) / n


def alineacion(act: torch.Tensor, decaimiento: float, alcance: int) -> torch.Tensor:
    """Penaliza que imagenes vecinas del lote activen distinto una capa.

    Es el "small alignment objective" de la interpolacion del articulo:
    empuja a que la imagen i y la i+d se parezcan en una capa BAJA, para que
    el mismo objeto se dibuje en el mismo sitio a lo largo de la fila. Las
    parejas mas alejadas pesan menos: 1 / decaimiento**d, con d de 1 a alcance.

    Devuelve una penalizacion (>= 0) que se SUMA a la perdida. lucid la
    devuelve con el signo cambiado porque alli todo se maximiza.

    Referencia: lucid.optvis.objectives.alignment.
    """
    total = act.new_zeros(())
    for d in range(1, alcance + 1):
        for i in range(act.shape[0] - d):
            total = total + (act[i] - act[i + d]).pow(2).mean() / decaimiento ** d
    return total


def _norma_l1(x: torch.Tensor) -> torch.Tensor:
    """Norma L1 de la imagen, el primero de los penalizadores de frecuencia.

    lucid usa reduce_sum; aqui se promedia, igual que tv y l2, para que los
    tres pesos esten en la misma escala entre si. La consecuencia es que los
    coeficientes del articulo (L1 -0.05, TV -0.25) NO se pueden copiar tal
    cual: alli van sobre sumas de 150.528 elementos a 224x224.
    """
    return x.abs().mean()


def _desenfocar(x: torch.Tensor) -> torch.Tensor:
    """Desenfoque 3x3 de lucid (_tf_blur), normalizado en los bordes.

    El nucleo es 0.5 en todas las casillas y 1.0 en el centro, y se divide
    por el mismo desenfoque aplicado a unos, de modo que en los bordes los
    pesos se reparten solo entre los vecinos que existen.
    """
    nucleo = torch.full((1, 1, 3, 3), 0.5, dtype=x.dtype, device=x.device)
    nucleo[0, 0, 1, 1] = 1.0
    nucleo = nucleo.expand(x.shape[1], 1, 3, 3)
    difuminar = lambda t: F.conv2d(t, nucleo, padding=1, groups=t.shape[1])
    return difuminar(x) / difuminar(torch.ones_like(x))


def _penalizacion_desenfoque(x: torch.Tensor) -> torch.Tensor:
    """Penaliza la diferencia con la version desenfocada de la imagen.

    Es blur_input_each_step de lucid: minimizarla equivale a desenfocar la
    imagen un poco en cada paso, que es como el articulo describe esta
    familia. El termino desenfocado va detached, igual que el stop_gradient
    de lucid: el gradiente solo debe empujar la imagen hacia el desenfoque,
    no mover el desenfoque hacia la imagen.
    """
    return 0.5 * (x - _desenfocar(x).detach()).pow(2).mean()


def _variacion_total(x: torch.Tensor) -> torch.Tensor:
    """Variacion total: penaliza el ruido de alta frecuencia que quede."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def cargar_config(ruta: Path) -> dict:
    """Lee el YAML de configuracion."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def nombre_run(ocfg: dict, args) -> str:
    """Como se llama el run en W&B.

    El objetivo, mas los mandos que se hayan cambiado por linea de comandos.
    Sin esto todos los runs se llaman igual (experimento.nombre) y una lista
    de veinte es ilegible, que es justo lo contrario de para lo que sirve
    registrar experimentos.

    Solo se anaden los mandos pasados por CLI: son los que distinguen este
    run de la configuracion del YAML, que ya se guarda entera aparte.
    """
    partes = [nombre_salida(ocfg)]
    for bandera, valor in (("pasos", args.pasos), ("lr", args.learning_rate),
                           ("l1", args.l1), ("tv", args.tv), ("blur", args.blur),
                           ("jitter", args.jitter), ("lote", args.lote)):
        if valor is not None:
            partes.append(f"{bandera}{valor:g}" if isinstance(valor, (int, float))
                          else f"{bandera}{valor}")
    if args.parametrizacion:
        partes.append(args.parametrizacion)
    if args.sin_robustez:
        partes.append("sin-robustez")
    if getattr(args, "diversidad", False):
        partes.append("diversidad")
    return "-".join(partes)


def nombre_salida(ocfg: dict) -> str:
    """Nombre de fichero que identifica el objetivo sin ambiguedad.

    Dos objetivos distintos sobre la misma capa y canal (Channel y Neuron) no
    pueden acabar en el mismo PNG, o uno pisa al otro sin avisar. Layer no
    elige canal, asi que no lo lleva: si lo llevara, el mismo DeepDream
    saldria con un nombre distinto segun lo que quedase en el YAML. Direction
    tampoco elige canal: lo que la identifica es la semilla de su vector.
    """
    capa = str(ocfg["capa"]).replace(".", "_")
    tipo = str(ocfg.get("tipo", "canal"))
    partes = [capa, tipo]
    if tipo == "direccion":
        partes.append(f"s{ocfg['direccion']}")
    elif tipo == "suma":
        partes.append("+".join(str(z) for z in ocfg["canales"]))
    elif tipo == "interpolacion":
        partes.append("-".join(str(z) for z in ocfg["canales"]))
    elif tipo != "capa":
        partes.append(str(ocfg["canal"]))
    if tipo == "neurona":
        x, y = ocfg.get("x"), ocfg.get("y")
        partes.append("centro" if x is None and y is None else f"{x}x{y}")
    return "_".join(partes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/feature_viz.yaml"))
    ap.add_argument("--tipo", choices=list(OBJETIVOS), default=None,
                    help="Channel (canal entero), Neuron (una posicion), "
                         "Layer (capa entera al cuadrado, DeepDream), "
                         "Class Logits (logit de la clase, solo en fc), "
                         "Class Probability (softmax de la clase, solo en fc), "
                         "Direction (vector aleatorio del espacio de activaciones), "
                         "suma de varios canales (--canales) o interpolacion "
                         "entre dos (--canales A B --lote N)")
    ap.add_argument("--capa", type=str, default=None,
                    help="modulo a maximizar, p.ej. 'fc' o 'layer3.1.conv2'")
    ap.add_argument("--canal", type=int, default=None,
                    help="indice z del canal, o la clase con --tipo logit "
                         "o probabilidad")
    ap.add_argument("--x", type=int, default=None,
                    help="fila del mapa; por defecto el centro (solo con --tipo neurona)")
    ap.add_argument("--y", type=int, default=None,
                    help="columna del mapa; por defecto el centro")
    ap.add_argument("--direccion", type=int, default=None,
                    help="semilla del vector aleatorio (solo con --tipo direccion)")
    ap.add_argument("--canales", type=int, nargs="+", default=None,
                    help="canales a sumar con --tipo suma, o de partida y de "
                         "llegada con --tipo interpolacion")
    ap.add_argument("--lote", type=int, default=None,
                    help="imagenes que se optimizan a la vez; la interpolacion "
                         "necesita 2 o mas")
    ap.add_argument("--diversidad", nargs="?", const="sugerido", default=None,
                    metavar="PESO",
                    help="activa el termino de diversidad (necesita --lote 2 o "
                         "mas); sin valor usa regularizacion.diversidad."
                         "peso_sugerido del YAML")
    ap.add_argument("--capa-diversidad", type=str, default=None,
                    help="capa donde se comparan las imagenes; por defecto la "
                         "del objetivo. Con objetivos en fc, una conv (layer4)")
    ap.add_argument("--pasos", type=int, default=None, help="pasos de optimizacion")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="tasa de aprendizaje de Adam")
    # Los controles de las figuras interactivas del articulo.
    ap.add_argument("--parametrizacion", choices=["fourier", "pixel"], default=None,
                    help="'pixel' apaga el preconditioning (Decorrelated Space)")
    ap.add_argument("--sin-robustez", action="store_true",
                    help="apaga la robustez a transformaciones")
    ap.add_argument("--l1", type=float, default=None, help="peso del penalizador L1")
    ap.add_argument("--tv", type=float, default=None, help="peso de la variacion total")
    ap.add_argument("--blur", type=float, default=None, help="peso del desenfoque")
    ap.add_argument("--jitter", type=int, default=None, help="desplazamiento maximo")
    ap.add_argument("--escalas", type=float, nargs="+", default=None,
                    help="factores de escala entre los que elegir")
    ap.add_argument("--rotaciones", type=float, nargs="+", default=None,
                    help="angulos en grados entre los que elegir")
    args = ap.parse_args()

    ruta = args.config if args.config.is_absolute() else RAIZ / args.config
    if not ruta.is_file():
        print(f"error: no existe {args.config}", file=sys.stderr)
        return 2
    cfg = cargar_config(ruta)

    # Solo se sobreescribe lo que se pide por CLI; el resto lo manda el YAML.
    for clave, valor in (("tipo", args.tipo), ("capa", args.capa),
                         ("canal", args.canal), ("x", args.x), ("y", args.y),
                         ("direccion", args.direccion), ("canales", args.canales)):
        if valor is not None:
            cfg["objetivo"][clave] = valor
    if args.pasos is not None:
        cfg["optimizacion"]["pasos"] = args.pasos
    if args.learning_rate is not None:
        cfg["optimizacion"]["learning_rate"] = args.learning_rate
    if args.lote is not None:
        cfg["imagen"]["lote"] = args.lote
    if args.diversidad is not None:
        dcfg = cfg["regularizacion"].setdefault("diversidad", {})
        dcfg["peso"] = float(dcfg["peso_sugerido"] if args.diversidad == "sugerido"
                             else args.diversidad)
    if args.capa_diversidad is not None:
        cfg["regularizacion"].setdefault("diversidad", {})["capa"] = args.capa_diversidad
    if args.parametrizacion is not None:
        cfg["imagen"]["parametrizacion"] = args.parametrizacion
    for clave, valor in (("l1", args.l1), ("tv", args.tv), ("blur", args.blur)):
        if valor is not None:
            cfg["regularizacion"][clave] = valor
    if args.sin_robustez:
        cfg["robustez"]["activa"] = False
    if args.jitter is not None:
        cfg["robustez"]["jitter"] = args.jitter
    if args.escalas is not None:
        cfg["robustez"]["escalas"] = args.escalas
    if args.rotaciones is not None:
        cfg["robustez"]["rotaciones"] = args.rotaciones

    fijar_seed(int(cfg["experimento"]["seed"]))
    run = abrir_run(cfg, nombre_run(cfg["objetivo"], args))

    imagen, historial = maximizar_activacion(cfg, run)

    destino = RAIZ / cfg["salida"]["directorio"]
    destino.mkdir(parents=True, exist_ok=True)
    nombre = nombre_salida(cfg["objetivo"])
    # Con diversidad sale otra cosa que sin ella: no puede pisar ese PNG.
    if float((cfg["regularizacion"].get("diversidad") or {}).get("peso", 0.0)) \
            and int(cfg["imagen"].get("lote", 1)) > 1:
        nombre += "_diversidad"
    fichero = destino / (nombre + ".png")
    # Un lote se guarda como UNA fila: es como se lee una interpolacion.
    save_image(imagen, fichero, nrow=imagen.shape[0])

    print(f"\nactivacion inicial: {historial[0]:+.4f}")
    print(f"activacion final  : {historial[-1]:+.4f}")
    print(f"aspereza final    : {_variacion_total(imagen).item():.3f}")
    print(f"imagen guardada en: {fichero.relative_to(RAIZ)}")

    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
