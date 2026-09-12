#!/usr/bin/env python3
"""Descarga y verificacion de los datos.

Los datos NO van al repositorio. Lo que va es este script: pequeno, versionado
y con el hash del fichero, de forma que cualquiera que clone obtenga
exactamente los mismos bytes que tu.

Es lo que hace posible la auditoria cruzada de noviembre: si tu auditor no
puede descargar tus datos, no puede reproducir tu metrica, y eso resta puntos
a las dos partes.

    python scripts/get_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DESTINO = Path("data")

# Rellena esto con tu dataset. El hash se obtiene la primera vez con
# `python scripts/get_data.py --hash`.
FUENTES = [
    # ("https://.../train.tar.gz", "train.tar.gz", "sha256:abc123..."),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path) -> None:
    print(f"descargando {url} -> {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destino)


def main() -> int:
    solo_hash = "--hash" in sys.argv
    if not FUENTES:
        print("No hay fuentes declaradas. Rellena FUENTES en este fichero.")
        print("Sin esto, nadie puede reproducir tu trabajo (hito H3).")
        return 1

    for url, nombre, esperado in FUENTES:
        ruta = DESTINO / nombre
        if not ruta.exists():
            descargar(url, ruta)

        real = sha256(ruta)
        if solo_hash:
            print(f"{nombre}: sha256:{real}")
            continue

        if esperado.removeprefix("sha256:") != real:
            print(f"ERROR: el hash de {nombre} no coincide.")
            print(f"  esperado: {esperado}")
            print(f"  obtenido: sha256:{real}")
            print("  El fichero esta corrupto o el origen ha cambiado.")
            return 1
        print(f"ok  {nombre}  sha256 verificado")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
