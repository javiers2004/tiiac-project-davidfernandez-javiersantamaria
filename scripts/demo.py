#!/usr/bin/env python3
"""Recorrido completo por terminal, para la demostracion en directo.

    python scripts/demo.py                       # elige una foto del test
    python scripts/demo.py mi_planta.jpg         # tu propia foto
    python scripts/demo.py --capa layer4.1.conv2 --pasos 128

Hace los cuatro pasos del articulo y los va narrando:

    1. que especie ve el modelo
    2. que neuronas enciende esa foto
    3. donde se encienden dentro de la foto
    4. que patron busca la mas activa

El paso 4 se registra en Weights & Biases, asi que al terminar wandb imprime
el enlace del run. Necesita la clave en el entorno:

    WANDB_API_KEY=...  python scripts/demo.py

Sin ella funciona igual, pero sin registrar.

Todo lo que imprime es ASCII a proposito: la consola de Windows usa cp1252 y
los caracteres de dibujo la hacen reventar a media demostracion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from torchvision.utils import save_image  # noqa: E402

from src.interpretability.api import (  # noqa: E402
    CONFIG,
    cargar_config,
    cargar_modelo,
    generar_imagen,
    listar_neuronas,
    mapa_activacion,
    predecir,
    superponer_mapa,
)


def titulo(n: int, texto: str) -> None:
    print(f"\n{n}. {texto}")
    print("-" * 62)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("foto", nargs="?", help="ruta a una foto; por defecto una del test")
    ap.add_argument("--capa", default="layer3.1.conv2")
    ap.add_argument("--canal", type=int, default=None,
                    help="salta el ranking y optimiza este canal")
    ap.add_argument("--pasos", type=int, default=256)
    ap.add_argument("--tipo", default="canal",
                    choices=["canal", "neurona", "logit", "probabilidad"])
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

    salida = RAIZ / "outputs" / "demo"
    salida.mkdir(parents=True, exist_ok=True)
    registrar = bool(os.environ.get("WANDB_API_KEY"))
    print(f"\nfoto: {foto.name}")
    if not registrar:
        print("aviso: sin WANDB_API_KEY, el run no se registrara")

    modelo = cargar_modelo()

    # ------------------------------------------------------------------ 1
    titulo(1, "Que especie ve el modelo")
    for i, p in enumerate(predecir(foto, top_k=3, modelo=modelo)):
        marca = "->" if i == 0 else "  "
        print(f"  {marca} {p.nombre or f'clase {p.clase}':<42} {p.probabilidad:6.1%}")

    # ------------------------------------------------------------------ 2
    titulo(2, f"Que neuronas enciende, en {args.capa}")
    top = listar_neuronas(foto, capas=args.capa, top_k=6, modelo=modelo)
    tope = max(abs(a.valor) for a in top) or 1
    for a in top:
        print(f"  canal {a.canal:>4}  {a.valor:+7.3f}  "
              f"{'#' * int(abs(a.valor) / tope * 28)}")

    elegida = top[0] if args.canal is None else next(
        (a for a in top if a.canal == args.canal), top[0])
    print(f"\n  se optimizara el canal {elegida.canal}")

    # ------------------------------------------------------------------ 3
    titulo(3, "Donde se enciende dentro de la foto")
    mapa = mapa_activacion(foto, elegida, modelo=modelo)
    destino = salida / f"mapa_{args.capa.replace('.', '_')}_{elegida.canal}.png"
    save_image(superponer_mapa(foto, mapa), destino)
    print(f"  mapa de calor -> {destino.relative_to(RAIZ)}")
    print("  (claro = la neurona se dispara ahi)")

    # ------------------------------------------------------------------ 4
    titulo(4, f"Que patron busca el canal {elegida.canal}")
    print(f"  partiendo de ruido, {args.pasos} pasos")
    if registrar:
        print("  registrando el run en W&B\n")
    else:
        print()

    # El propio bucle va imprimiendo "paso N/M activacion=...", asi que aqui
    # no se anade otra barra: dos cosas escribiendo a la vez en la misma
    # linea se pisan y la salida queda ilegible.
    t0 = time.perf_counter()
    imagen = generar_imagen(elegida, tipo=args.tipo, pasos=args.pasos,
                            modelo=modelo, registrar=registrar)

    patron = salida / f"patron_{args.capa.replace('.', '_')}_{elegida.canal}.png"
    save_image(imagen, patron)
    print(f"\n  {time.perf_counter() - t0:.0f} s")
    print(f"  patron -> {patron.relative_to(RAIZ)}")
    if registrar:
        # wandb imprime el enlace directo del run, pero solo cuando la
        # salida es un terminal de verdad. Por si acaso, el del proyecto:
        w = cargar_config(CONFIG)["wandb"]
        print('  W&B -> https://wandb.ai/'
              + w['entidad'] + '/' + w['proyecto'])
    print("\n  Los pesos del modelo no se han tocado: lo que se ha optimizado")
    print("  es la imagen, desde ruido aleatorio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
