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
from typing import Any

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
    construir_objetivo,
    inspeccionar_capa,
    resolver_modulo,
)
from src.utils.seed import fijar_seed               # noqa: E402
from src.utils.tracking import abrir_run, registrar  # noqa: E402
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
                         objetivo: Any | None = None
                         ) -> tuple[torch.Tensor, list[float]]:
    """Optimiza una imagen de ruido para maximizar la activacion de una neurona.

    Args:
        cfg: configuracion completa, tal cual sale del YAML.
        run: run de W&B abierto, o None.
        modelo: modelo ya congelado. Si es None se carga el del checkpoint.
            Se inyecta desde los tests, que no tienen los .tar disponibles.

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

    tam = int(icfg["tamano"])
    parametrizacion = ImagenFourier(tam, tam, decay=float(icfg["decay_power"]),
                                    sd=float(icfg["sd_inicial"]),
                                    generador=generador,
                                    lote=int(icfg.get("lote", 1)),
                                    compartida=icfg.get("compartida")).to(dispositivo)

    rcfg, gcfg = cfg["robustez"], cfg["regularizacion"]
    peso_tv, peso_l2 = float(gcfg["tv"]), float(gcfg["l2"])
    acfg = gcfg.get("alineacion") or {}
    peso_alin = float(acfg.get("peso", 0.0))
    # La alineacion compara imagenes del lote entre si: con una sola seria
    # cero, asi que ni se calcula. Por eso puede tener peso en el YAML sin
    # afectar a los objetivos de una imagen.
    if parametrizacion.lote < 2:
        peso_alin = 0.0
    capas_alin = [str(c) for c in acfg["capas"]] if peso_alin else []

    # Los hooks capturan la salida de cada capa que hace falta: la del
    # objetivo y, si hay alineacion, las suyas (puede incluir la misma).
    # Sirven igual para 'fc' (2D, N x clases) que para una conv (4D).
    capturado: dict[str, torch.Tensor] = {}
    asas = [
        resolver_modulo(modelo, c).register_forward_hook(
            lambda _m, _entrada, salida, c=c: capturado.__setitem__(c, salida))
        for c in dict.fromkeys([capa, *capas_alin])
    ]

    optimizador = torch.optim.Adam(parametrizacion.parameters(),
                                   lr=float(pcfg["learning_rate"]))
    ncfg = cfg["normalizacion"]
    media = torch.tensor(ncfg["media"], device=dispositivo).view(1, 3, 1, 1)
    desv = torch.tensor(ncfg["desviacion"], device=dispositivo).view(1, 3, 1, 1)

    pasos = int(pcfg["pasos"])
    historial: list[float] = []
    try:
        for paso in range(pasos):
            optimizador.zero_grad(set_to_none=True)

            imagen = parametrizacion()
            entrada = aplicar_robustez(imagen, rcfg, generador)
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
            # la imagen y taparian una unidad muerta.
            if paso == 0:
                _comprobar_gradiente(parametrizacion, activacion, objetivo, capa)

            perdida = -activacion
            metricas = {"viz/activacion": activacion.item()}
            if peso_tv:
                perdida = perdida + peso_tv * _variacion_total(imagen)
            if peso_l2:
                perdida = perdida + peso_l2 * imagen.pow(2).mean()
            if peso_alin:
                penal = sum(alineacion(capturado[c], float(acfg["decaimiento"]),
                                       int(acfg["alcance"])) for c in capas_alin)
                perdida = perdida + peso_alin * penal
                metricas["viz/alineacion"] = penal.item()

            perdida.backward()

            optimizador.step()

            # .item() rompe el grafo: sin esto se acumula memoria.
            valor = metricas["viz/activacion"]
            historial.append(valor)
            registrar(run, {**metricas, "viz/perdida": perdida.item()}, paso)
            if paso % max(1, pasos // 10) == 0:
                print(f"paso {paso:4d}/{pasos}  activacion={valor:+.4f}")
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
    """Falla pronto si el objetivo no mueve la imagen."""
    grad, = torch.autograd.grad(activacion, parametrizacion.espectro,
                                retain_graph=True, allow_unused=True)
    if grad is not None and torch.any(grad != 0):
        return
    raise UnidadMuerta(
        f"{objetivo} en '{capa}' no propaga gradiente: la unidad esta muerta "
        f"(un ReLU la deja en cero) y la imagen no cambiaria.\n"
        f"Opciones: apuntar a la convolucion PRE-ReLU en vez de a la salida "
        f"del bloque (p.ej. '{capa}.1.conv2' en lugar de '{capa}'), elegir "
        f"otro canal, o mover (x, y) a otra posicion."
    )


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


def _variacion_total(x: torch.Tensor) -> torch.Tensor:
    """Variacion total: penaliza el ruido de alta frecuencia que quede."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def cargar_config(ruta: Path) -> dict:
    """Lee el YAML de configuracion."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    ap.add_argument("--pasos", type=int, default=None, help="pasos de optimizacion")
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
    if args.lote is not None:
        cfg["imagen"]["lote"] = args.lote

    fijar_seed(int(cfg["experimento"]["seed"]))
    run = abrir_run(cfg)

    imagen, historial = maximizar_activacion(cfg, run)

    destino = RAIZ / cfg["salida"]["directorio"]
    destino.mkdir(parents=True, exist_ok=True)
    fichero = destino / (nombre_salida(cfg["objetivo"]) + ".png")
    # Un lote se guarda como UNA fila: es como se lee una interpolacion.
    save_image(imagen, fichero, nrow=imagen.shape[0])

    print(f"\nactivacion inicial: {historial[0]:+.4f}")
    print(f"activacion final  : {historial[-1]:+.4f}")
    print(f"imagen guardada en: {fichero.relative_to(RAIZ)}")

    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
