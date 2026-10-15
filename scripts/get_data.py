#!/usr/bin/env python3
"""Descarga y verificacion de Pl@ntNet-300K.

Los datos NO van al repositorio. Lo que va es este script, con el checksum del
fichero, de forma que cualquiera que clone obtenga exactamente los mismos
bytes.

    python scripts/get_data.py --solo-metadatos   # ~70 MB, en un minuto
    python scripts/get_data.py                    # el dataset entero, 31.7 GB

Los metadatos van aparte a proposito. Pesan 70 MB frente a 31.7 GB y son todo
lo que hace falta para traducir la salida del modelo a nombres de especie, asi
que la aplicacion se puede desarrollar sin tener el dataset. Se extraen del
zip remoto con peticiones HTTP por rango: un zip guarda su indice al final, de
modo que se puede leer ese indice y bajar solo los bytes de los miembros que
interesan.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data"
METADATOS = DESTINO / "metadata"

# Zenodo record 5645731, version 1.1. La 1.0 trae las mismas imagenes pero
# distintos metadatos; el repo oficial recomienda la 1.1.
URL = "https://zenodo.org/api/records/5645731/files/plantnet_300K.zip/content"
NOMBRE = "plantnet_300K.zip"
MD5 = "db27d149f2a6c304b887353c07021687"   # publicado por Zenodo

BLOQUE = 4 << 20


class _FicheroHttp(io.RawIOBase):
    """Fichero remoto de solo lectura, servido por peticiones de rango."""

    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        peticion = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(peticion) as r:
            self.size = int(r.headers["Content-Length"])

    def _bloque(self, indice: int) -> bytes:
        if indice not in self.cache:
            inicio = indice * BLOQUE
            fin = min(inicio + BLOQUE, self.size) - 1
            peticion = urllib.request.Request(
                self.url, headers={"Range": f"bytes={inicio}-{fin}"})
            with urllib.request.urlopen(peticion) as r:
                self.cache[indice] = r.read()
        return self.cache[indice]

    def readinto(self, b) -> int:
        # RawIOBase enruta read() por aqui; sin esto da NotImplementedError.
        n = min(len(b), self.size - self.pos)
        escrito = 0
        while escrito < n:
            idx, off = divmod(self.pos, BLOQUE)
            trozo = self._bloque(idx)[off:off + (n - escrito)]
            if not trozo:
                break
            b[escrito:escrito + len(trozo)] = trozo
            self.pos += len(trozo)
            escrito += len(trozo)
        return escrito

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}
        self.pos = max(0, min(base[whence] + offset, self.size))
        return self.pos

    def tell(self) -> int: return self.pos
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True


def md5(ruta: Path) -> str:
    """MD5 del fichero. Zenodo publica MD5, no SHA-256."""
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar_metadatos() -> None:
    """Saca los JSON de metadatos del zip remoto sin bajarlo entero."""
    METADATOS.mkdir(parents=True, exist_ok=True)
    print("leyendo el indice del zip remoto (unos 60 MB)...")

    remoto = _FicheroHttp(URL)
    with zipfile.ZipFile(io.BufferedReader(remoto, BLOQUE)) as z:
        for nombre in (n for n in z.namelist() if n.endswith(".json")):
            salida = METADATOS / Path(nombre).name
            salida.write_bytes(z.read(nombre))
            print(f"  ok  {salida.relative_to(RAIZ)}  ({salida.stat().st_size:,} B)")

    _derivar_mapa_de_clases()


def _derivar_mapa_de_clases() -> None:
    """Reconstruye class_idx_to_species_id.json.

    El repo oficial lo documenta pero NO lo publica: no esta en el zip de
    Zenodo, ni en el repositorio, ni en sus releases. Se deriva de como
    torchvision construye las clases: ImageFolder ordena alfabeticamente los
    nombres de carpeta, y en este dataset las carpetas son los ids de especie.
    Es exactamente el class_to_idx que uso el entrenamiento original, porque
    src/utils/utils.py monta Plantnet(ImageFolder) sobre root/train.

    Ojo: el orden es de CADENAS, no numerico, que es lo que hace sorted()
    sobre nombres de directorio.
    """
    origen = METADATOS / "plantnet300K_metadata.json"
    if not origen.is_file():
        print("aviso: falta plantnet300K_metadata.json; no se deriva el mapa")
        return

    meta = json.loads(origen.read_text(encoding="utf-8"))
    especies = sorted({v["species_id"] for v in meta.values()
                       if v["split"] == "train"})
    if len(especies) != 1081:
        print(f"aviso: salen {len(especies)} especies y se esperaban 1081")

    destino = METADATOS / "class_idx_to_species_id.json"
    destino.write_text(json.dumps({str(i): s for i, s in enumerate(especies)},
                                  indent=1), encoding="utf-8")
    print(f"  ok  {destino.relative_to(RAIZ)}  (derivado, {len(especies)} especies)")


def _especies_frecuentes(minimo: int = 300) -> set[str]:
    """Especies con al menos `minimo` imagenes de entrenamiento.

    Devuelve un conjunto vacio si los metadatos no estan: entonces no se
    filtra y se avisa.
    """
    origen = METADATOS / "plantnet300K_metadata.json"
    if not origen.is_file():
        return set()

    from collections import Counter
    meta = json.loads(origen.read_text(encoding="utf-8"))
    cuenta = Counter(v["species_id"] for v in meta.values()
                     if v["split"] == "train")
    return {s for s, n in cuenta.items() if n >= minimo}


def descargar_fotos(cuantas: int = 5) -> None:
    """Saca unas pocas fotos del split de TEST, cada una de otra especie.

    Para probar la aplicacion no hacen falta 31.7 GB ni los 31.112 imagenes
    del test: con un punado basta. Se usa el mismo mecanismo de rangos, asi
    que esto baja unos pocos cientos de KB de imagen.
    """
    destino = DESTINO / "muestras"
    destino.mkdir(parents=True, exist_ok=True)

    # Pl@ntNet-300K es de cola larga: la mediana es de 21 imagenes de train
    # por especie y 523 de las 1081 tienen menos de 20. Si se cogen especies
    # al azar casi siempre tocan raras, el modelo falla y parece roto cuando
    # lo que pasa es que apenas las ha visto. Para una demo se eligen especies
    # bien representadas.
    comunes = _especies_frecuentes()
    print(f"buscando {cuantas} fotos de test"
          + (f" entre las {len(comunes)} especies mas frecuentes..." if comunes
             else " (sin metadatos: no se puede filtrar por frecuencia)..."))

    remoto = _FicheroHttp(URL)
    with zipfile.ZipFile(io.BufferedReader(remoto, BLOQUE)) as z:
        vistas: set[str] = set()
        elegidas: list[str] = []
        for nombre in z.namelist():
            partes = Path(nombre).parts
            if "test" not in partes or not nombre.lower().endswith(".jpg"):
                continue
            especie = partes[-2]
            if especie in vistas or (comunes and especie not in comunes):
                continue
            vistas.add(especie)
            elegidas.append(nombre)
            if len(elegidas) >= cuantas:
                break

        for nombre in elegidas:
            especie = Path(nombre).parts[-2]
            salida = destino / f"{especie}_{Path(nombre).name}"
            salida.write_bytes(z.read(nombre))
            print(f"  ok  {salida.relative_to(RAIZ)}  ({salida.stat().st_size:,} B)")


def descargar_dataset() -> int:
    """Baja el zip entero y verifica el MD5."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    ruta = DESTINO / NOMBRE

    if not ruta.exists():
        print(f"descargando {NOMBRE} (31.7 GB)...")
        urllib.request.urlretrieve(URL, ruta)

    print("verificando el checksum...")
    real = md5(ruta)
    if real != MD5:
        print(f"ERROR: el MD5 de {NOMBRE} no coincide.")
        print(f"  esperado: {MD5}")
        print(f"  obtenido: {real}")
        print("  El fichero esta corrupto o el origen ha cambiado.")
        return 1

    print(f"ok  {NOMBRE}  md5 verificado")
    print(f"Descomprimelo en {DESTINO}/ para obtener train/, val/ y test/.")
    return 0


def main() -> int:
    if "--fotos" in sys.argv:
        i = sys.argv.index("--fotos")
        cuantas = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 5
        descargar_fotos(cuantas)
        return 0

    if "--solo-metadatos" in sys.argv:
        descargar_metadatos()
        return 0

    codigo = descargar_dataset()
    if codigo == 0:
        descargar_metadatos()
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
