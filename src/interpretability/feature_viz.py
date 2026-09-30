"""Visualizacion de caracteristicas por maximizacion de activacion.

Elige una neurona, parte de ruido y optimiza LA IMAGEN DE ENTRADA (no los
pesos) para maximizar la activacion de esa neurona. El resultado es el patron
que esa neurona busca.

    python -m src.interpretability.feature_viz --neurona 0
    python -m src.interpretability.feature_viz --capa layer3.1.conv2 --neurona 42

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
  3. Se maximiza el LOGIT, no la softmax. Maximizar la probabilidad premia
     hundir las otras 1080 clases, que es mas facil que representar la propia.

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
    """

    def __init__(self, alto: int, ancho: int, decay: float = 1.0,
                 sd: float = 0.01, generador: torch.Generator | None = None):
        super().__init__()
        self.alto, self.ancho = alto, ancho
        ruido = torch.randn(1, 3, alto, ancho // 2 + 1, 2, generator=generador)
        self.espectro = nn.Parameter(ruido * sd)
        self.register_buffer("escala", self._escala_frecuencias(alto, ancho, decay))

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
        """Reconstruye la imagen RGB en [0, 1] a partir del espectro."""
        espectro = torch.view_as_complex(self.espectro) * self.escala
        imagen = torch.fft.irfft2(espectro, s=(self.alto, self.ancho))
        imagen = imagen / 4.0  # constante empirica de Olah et al.

        # Decorrelacion de color: (B,3,H,W) -> (N,3) -> matmul -> vuelta.
        color = torch.as_tensor(_COLOR_NORM, device=imagen.device)
        plano = imagen.permute(0, 2, 3, 1).reshape(-1, 3) @ color.T
        imagen = plano.view(1, self.alto, self.ancho, 3).permute(0, 3, 1, 2)

        # La sigmoide mantiene la imagen en [0,1] sin matar el gradiente, que
        # es lo que haria un clamp en cuanto la imagen tocase los extremos.
        return torch.sigmoid(imagen)


def aplicar_robustez(imagen: torch.Tensor, rcfg: dict,
                     generador: torch.Generator) -> torch.Tensor:
    """Aplica jitter, escala y rotacion aleatorios a la imagen.

    Args:
        imagen: tensor (1, 3, H, W) en [0, 1].
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


def resolver_modulo(modelo: nn.Module, ruta: str) -> nn.Module:
    """Devuelve el submodulo indicado por una ruta con puntos ('layer3.1.conv2')."""
    modulo: nn.Module = modelo
    for parte in ruta.split("."):
        modulo = modulo[int(parte)] if parte.isdigit() else getattr(modulo, parte)
    return modulo


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
                         modelo: nn.Module | None = None
                         ) -> tuple[torch.Tensor, list[float]]:
    """Optimiza una imagen de ruido para maximizar la activacion de una neurona.

    Args:
        cfg: configuracion completa, tal cual sale del YAML.
        run: run de W&B abierto, o None.
        modelo: modelo ya congelado. Si es None se carga el del checkpoint.
            Se inyecta desde los tests, que no tienen los .tar disponibles.

    Returns:
        (imagen, historial) con la imagen final (1,3,H,W) en [0,1] y la
        activacion alcanzada en cada paso.
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if modelo is None:
        modelo = construir_modelo_congelado(cfg["modelo"])
    modelo = modelo.to(dispositivo)

    ocfg, icfg, pcfg = cfg["objetivo"], cfg["imagen"], cfg["optimizacion"]
    capa, neurona = str(ocfg["capa"]), int(ocfg["neurona"])

    # Un generador propio: las semillas globales las fija src/utils/seed.py y
    # no se tocan desde aqui.
    generador = torch.Generator().manual_seed(int(cfg["experimento"]["seed"]))

    tam = int(icfg["tamano"])
    parametrizacion = ImagenFourier(tam, tam, decay=float(icfg["decay_power"]),
                                    sd=float(icfg["sd_inicial"]),
                                    generador=generador).to(dispositivo)

    # El hook captura la salida de la capa elegida. Sirve igual para 'fc'
    # (salida 2D, N x clases) que para una conv intermedia (4D, N x C x H x W).
    capturado: dict[str, torch.Tensor] = {}
    modulo = resolver_modulo(modelo, capa)
    asa = modulo.register_forward_hook(
        lambda _m, _entrada, salida: capturado.__setitem__("act", salida)
    )

    optimizador = torch.optim.Adam(parametrizacion.parameters(),
                                   lr=float(pcfg["learning_rate"]))
    ncfg = cfg["normalizacion"]
    media = torch.tensor(ncfg["media"], device=dispositivo).view(1, 3, 1, 1)
    desv = torch.tensor(ncfg["desviacion"], device=dispositivo).view(1, 3, 1, 1)
    rcfg, gcfg = cfg["robustez"], cfg["regularizacion"]
    peso_tv, peso_l2 = float(gcfg["tv"]), float(gcfg["l2"])

    pasos = int(pcfg["pasos"])
    historial: list[float] = []
    try:
        for paso in range(pasos):
            optimizador.zero_grad(set_to_none=True)

            imagen = parametrizacion()
            entrada = aplicar_robustez(imagen, rcfg, generador)
            modelo((entrada - media) / desv)

            act = capturado["act"]
            if neurona >= act.shape[1]:
                raise IndexError(f"la capa '{capa}' tiene {act.shape[1]} unidades; "
                                 f"la neurona {neurona} no existe")
            # Para una conv, la media espacial del canal. Para fc, el logit.
            activacion = act[:, neurona].mean()

            perdida = -activacion
            if peso_tv:
                perdida = perdida + peso_tv * _variacion_total(imagen)
            if peso_l2:
                perdida = perdida + peso_l2 * imagen.pow(2).mean()

            perdida.backward()
            optimizador.step()

            # .item() rompe el grafo: sin esto se acumula memoria.
            valor = activacion.item()
            historial.append(valor)
            registrar(run, {"viz/activacion": valor,
                            "viz/perdida": perdida.item()}, paso)
            if paso % max(1, pasos // 10) == 0:
                print(f"paso {paso:4d}/{pasos}  activacion={valor:+.4f}")
    finally:
        asa.remove()

    with torch.no_grad():
        final = parametrizacion().detach().cpu()
    return final, historial


def _variacion_total(x: torch.Tensor) -> torch.Tensor:
    """Variacion total: penaliza el ruido de alta frecuencia que quede."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def cargar_config(ruta: Path) -> dict:
    """Lee el YAML de configuracion."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/feature_viz.yaml"))
    ap.add_argument("--capa", type=str, default=None,
                    help="modulo a maximizar, p.ej. 'fc' o 'layer3.1.conv2'")
    ap.add_argument("--neurona", type=int, default=None, help="indice de la neurona")
    ap.add_argument("--pasos", type=int, default=None, help="pasos de optimizacion")
    args = ap.parse_args()

    ruta = args.config if args.config.is_absolute() else RAIZ / args.config
    if not ruta.is_file():
        print(f"error: no existe {args.config}", file=sys.stderr)
        return 2
    cfg = cargar_config(ruta)

    # Solo se sobreescribe lo que se pide por CLI; el resto lo manda el YAML.
    if args.capa is not None:
        cfg["objetivo"]["capa"] = args.capa
    if args.neurona is not None:
        cfg["objetivo"]["neurona"] = args.neurona
    if args.pasos is not None:
        cfg["optimizacion"]["pasos"] = args.pasos

    fijar_seed(int(cfg["experimento"]["seed"]))
    run = abrir_run(cfg)

    imagen, historial = maximizar_activacion(cfg, run)

    destino = RAIZ / cfg["salida"]["directorio"]
    destino.mkdir(parents=True, exist_ok=True)
    capa = str(cfg["objetivo"]["capa"]).replace(".", "_")
    fichero = destino / f"{capa}_{cfg['objetivo']['neurona']}.png"
    save_image(imagen, fichero)

    print(f"\nactivacion inicial: {historial[0]:+.4f}")
    print(f"activacion final  : {historial[-1]:+.4f}")
    print(f"imagen guardada en: {fichero.relative_to(RAIZ)}")

    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
