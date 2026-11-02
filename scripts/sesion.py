#!/usr/bin/env python3
"""Recorrido paso a paso por la API, pensado para ensenarlo en directo.

    python scripts/sesion.py                       # una foto del test
    python scripts/sesion.py mi_planta.jpg
    python scripts/sesion.py --capa layer3.1.conv2 --pasos 128

A diferencia de scripts/demo.py, que hace todo el recorrido de una vez, aqui
cada llamada a la API se escribe en pantalla como si se tecleara en el
interprete y despues se ejecuta de verdad. Sirve para que se vea QUE funcion
se llama en cada etapa y con que argumentos, que es lo que no se aprecia
cuando un solo comando lo hace todo por dentro.

El modelo se carga una sola vez y se reutiliza en todos los pasos.

El paso de optimizacion se registra en Weights & Biases si hay clave:

    WANDB_API_KEY=...  python scripts/sesion.py

Todo lo que imprime es ASCII: la consola de Windows usa cp1252 y los
caracteres de dibujo la hacen reventar a media demostracion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from pprint import pformat
from typing import Any

# El aviso de torch.load ocupa media pantalla y aqui solo estorba.
warnings.filterwarnings("ignore", category=FutureWarning)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from torchvision.utils import save_image  # noqa: E402

from src.interpretability import api  # noqa: E402

PAUSA_TECLA = 0.022
PROMPT = ">>> "
CONTINUA = "... "
ANCHO = 104


def _activar_colores() -> bool:
    """Enciende las secuencias ANSI en la consola de Windows.

    Returns:
        True si la consola ya entiende los codigos de color.
    """
    if os.name != "nt":
        return True
    import ctypes

    k = ctypes.windll.kernel32
    modo = ctypes.c_uint32()
    if not k.GetConsoleMode(k.GetStdHandle(-11), ctypes.byref(modo)):
        return False
    # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    return bool(k.SetConsoleMode(k.GetStdHandle(-11), modo.value | 0x0004))


COLOR = _activar_colores()


def _c(texto: str, codigo: str) -> str:
    """Envuelve el texto en un color ANSI, si la consola los entiende."""
    return f"\033[{codigo}m{texto}\033[0m" if COLOR else texto


def teclear(lineas: list[str]) -> None:
    """Escribe las lineas caracter a caracter, como si se tecleasen.

    La primera lleva el prompt del interprete y las demas el de
    continuacion. Esos prompts son adorno: no forman parte del codigo
    que se ejecuta.

    Args:
        lineas: las lineas de codigo a mostrar.
    """
    sys.stdout.write("\n")
    for i, linea in enumerate(lineas):
        sys.stdout.write(_c(PROMPT if i == 0 else CONTINUA, "92"))
        for ch in linea:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(PAUSA_TECLA)
        sys.stdout.write("\n")
    sys.stdout.flush()


def paso(codigo: str | list[str], entorno: dict[str, Any],
         espera: float = 1.2, mostrar: bool = True) -> None:
    """Muestra una llamada, la ejecuta y escribe lo que devuelve.

    Si el codigo es una expresion se evalua y se imprime su valor, igual
    que en el interprete; si es una asignacion se ejecuta y no imprime
    nada. Las listas se parten por lineas para que quepan en pantalla.

    Args:
        codigo: la linea, o las lineas, de Python a ejecutar.
        entorno: el espacio de nombres compartido entre todos los pasos.
        espera: segundos de pausa al terminar, para que de tiempo a leerlo.
        mostrar: False para no imprimir el valor devuelto.
    """
    lineas = [codigo] if isinstance(codigo, str) else list(codigo)
    teclear(lineas)
    fuente = "\n".join(lineas)
    try:
        valor = eval(fuente, entorno)          # noqa: S307  es codigo propio
        if mostrar and valor is not None:
            print(pformat(valor, width=ANCHO))
    except SyntaxError:
        exec(fuente, entorno)                  # noqa: S102  es codigo propio
    time.sleep(espera)


def comentario(texto: str) -> None:
    """Escribe una linea de comentario entre dos pasos."""
    print("\n" + _c(f"# {texto}", "90"))
    time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("foto", nargs="?",
                    help="ruta a una foto; por defecto, una del test")
    ap.add_argument("--capa", default="layer4.1.conv2")
    ap.add_argument("--pasos", type=int, default=256)
    args = ap.parse_args()

    if args.foto:
        foto = Path(args.foto)
    else:
        muestras = sorted((RAIZ / "data" / "muestras").glob("*.jpg"))
        if not muestras:
            print("error: no hay fotos en data/muestras/.\n"
                  "       python scripts/get_data.py --fotos 5", file=sys.stderr)
            return 2
        foto = muestras[0]
    if not foto.is_file():
        print(f"error: no existe {foto}", file=sys.stderr)
        return 2

    (RAIZ / "outputs" / "demo").mkdir(parents=True, exist_ok=True)
    registrar = bool(os.environ.get("WANDB_API_KEY"))

    # En pantalla, ruta relativa: cabe mejor y no ensena el disco entero.
    rel = (foto.relative_to(RAIZ).as_posix() if foto.is_relative_to(RAIZ)
           else foto.as_posix())
    entorno: dict[str, Any] = {
        "api": api, "save_image": save_image, "foto": rel,
        "capa": args.capa, "pasos": args.pasos, "registrar": registrar,
    }

    print(_c("  Feature Visualization (Olah et al., 2017)", "96"))
    print(_c("  ResNet-18 sobre Pl@ntNet-300K  --  recorrido por la API", "90"))
    if not registrar:
        print(_c("  aviso: sin WANDB_API_KEY, el run no se registrara", "93"))
    time.sleep(2.0)

    comentario("1. el modelo, con los pesos congelados: solo se optimiza la imagen")
    paso("modelo = api.cargar_modelo()", entorno, espera=1.0)
    paso("sum(p.numel() for p in modelo.parameters())", entorno)

    comentario("2. que especie ve el modelo en esta foto")
    paso("api.predecir(foto, top_k=3, modelo=modelo)", entorno, espera=4.0)

    comentario(f"3. que canales de {args.capa} enciende esa foto")
    paso("neuronas = api.listar_neuronas(foto, capas=capa, top_k=6, modelo=modelo)",
         entorno, espera=0.8)
    paso("neuronas", entorno, espera=4.0)
    paso("elegida = neuronas[0]", entorno, espera=0.8)

    comentario("4. donde se enciende dentro de la foto")
    paso("mapa = api.mapa_activacion(foto, elegida, modelo=modelo)",
         entorno, espera=0.8)
    paso("mapa.shape", entorno, espera=1.5)
    paso("save_image(api.superponer_mapa(foto, mapa), 'outputs/demo/mapa.png')",
         entorno, espera=2.0)

    comentario("5. la optimizacion: partiendo de ruido, sube la activacion de")
    comentario("   ese canal cambiando SOLO los pixeles de la imagen")
    paso(["imagen = api.generar_imagen(elegida, tipo='canal', pasos=pasos,",
          "                            modelo=modelo, registrar=registrar)"],
         entorno, espera=1.0)
    paso("imagen.shape", entorno, espera=1.5)
    paso("save_image(imagen, 'outputs/demo/patron.png')", entorno, espera=2.0)

    print()
    print(_c("  Los pesos del modelo no se han tocado en ningun paso.", "96"))
    print(_c("  Lo unico que se ha optimizado es la imagen.", "96"))
    time.sleep(2.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
