"""Comprobacion rapida de que los pesos preentrenados de Pl@ntNet-300K cargan.

    python src/main.py     # desde la raiz del repo
    python -m src.main     # equivalente

No entrena ni evalua: construye la arquitectura, le mete el checkpoint y
confirma que el forward produce las formas esperadas. Es la prueba de que el
contrato del checkpoint se cumple antes de montar nada encima.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision.models import resnet18

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.utils.utils import load_model  # noqa: E402

N_CLASES = 1081  # Pl@ntNet-300K
CHECKPOINT = RAIZ / "models" / "resnet18_weights_best_acc.tar"


def main() -> int:
    """Carga el checkpoint preentrenado y comprueba que el forward funciona."""
    if not CHECKPOINT.is_file():
        print(f"error: no existe {CHECKPOINT.relative_to(RAIZ)}", file=sys.stderr)
        return 2

    # El wheel de torch que fija requirements.txt es solo CPU en Windows, asi
    # que la GPU se detecta, no se da por supuesta.
    use_gpu = torch.cuda.is_available()

    modelo = resnet18(num_classes=N_CLASES)
    epoch = load_model(modelo, filename=str(CHECKPOINT), use_gpu=use_gpu)

    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.randn(1, 3, 224, 224))

    print(f"checkpoint : {CHECKPOINT.relative_to(RAIZ)}")
    print(f"dispositivo: {'cuda:0' if use_gpu else 'cpu'}")
    print(f"epoch      : {epoch}")
    print(f"salida     : {tuple(salida.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
