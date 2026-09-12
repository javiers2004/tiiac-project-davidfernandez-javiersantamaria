#!/usr/bin/env python3
"""
tiiac-check · comprobador de reproducibilidad de TIIAC
======================================================

Se ejecuta sobre un clon limpio del repositorio de una pareja y comprueba las
siete condiciones que la asignatura exige. Devuelve una puntuacion de 0 a 100 y
un informe legible.

    python tiiac_check.py                 # sobre el directorio actual
    python tiiac_check.py ../mi-proyecto  # sobre otro directorio
    python tiiac_check.py --no-smoke      # salta la ejecucion (rapido)
    python tiiac_check.py --json          # salida para CI

Codigo de salida 0 si pasa (>= 80 y ningun bloque critico a cero), 1 si no.

No depende de nada fuera de la biblioteca estandar, a proposito: tiene que
poder ejecutarse en cualquier maquina y en integracion continua sin instalar.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SMOKE_LIMITE_S = 300  # regla 5/20: el smoke test tiene que caber en 5 minutos
APROBADO = 80


# --------------------------------------------------------------------------
# infraestructura de comprobaciones
# --------------------------------------------------------------------------

class Resultado:
    def __init__(self, clave: str, titulo: str, max_pts: int, critico: bool = False):
        self.clave = clave
        self.titulo = titulo
        self.max_pts = max_pts
        self.critico = critico
        self.pts = 0
        self.ok: list[str] = []
        self.mal: list[str] = []
        self.aviso: list[str] = []

    def bien(self, msg: str, pts: int = 0):
        self.ok.append(msg)
        self.pts += pts

    def fallo(self, msg: str):
        self.mal.append(msg)

    def ojo(self, msg: str):
        self.aviso.append(msg)

    def cerrar(self):
        self.pts = max(0, min(self.pts, self.max_pts))
        return self


def lee(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def versionados(raiz: Path):
    """Ficheros que git tiene registrados, o None si no es un repositorio.

    Se usa en vez de recorrer el disco porque lo que importa es que hay EN el
    repositorio, no que hay en la carpeta. Un .venv con pesos de PyTorch dentro
    no es un problema: no esta versionado.
    """
    try:
        pr = subprocess.run(["git", "--no-optional-locks", "-C", str(raiz), "ls-files"],
                            capture_output=True, text=True, timeout=10,
                            encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if pr.returncode != 0:
        return None
    return [l.strip() for l in (pr.stdout or "").splitlines() if l.strip()]


def ficheros_py(raiz: Path) -> list[Path]:
    saltar = {".git", "venv", ".venv", "__pycache__", "wandb", "data",
              ".mypy_cache", "node_modules", ".ipynb_checkpoints"}
    out = []
    for p in raiz.rglob("*.py"):
        if any(part in saltar for part in p.parts):
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------
# 1 · estructura minima
# --------------------------------------------------------------------------

def check_estructura(raiz: Path) -> Resultado:
    r = Resultado("estructura", "Estructura minima del repositorio", 15, critico=True)

    obligatorios = {
        "README.md": 3,
        "requirements.txt": 3,
        "run.py": 4,
        ".gitignore": 2,
    }
    for nombre, pts in obligatorios.items():
        if (raiz / nombre).is_file():
            r.bien(f"{nombre} presente", pts)
        else:
            r.fallo(f"falta {nombre}")

    if (raiz / "src").is_dir() and ficheros_py(raiz / "src"):
        r.bien("src/ con codigo", 3)
    else:
        r.fallo("falta src/ con codigo fuente (no todo puede vivir en run.py)")

    # Basura versionada. Se mira SOLO lo que git tiene registrado: si esta
    # ignorado o vive en .venv, no esta en el repositorio y no es problema.
    seguidos = versionados(raiz)
    if seguidos is None:
        r.ojo("no es un repositorio git, asi que no se puede comprobar que hay versionado")
    else:
        pesados = [f for f in seguidos
                   if f.rsplit(".", 1)[-1].lower() in {"pt", "pth", "ckpt", "h5", "onnx"}]
        cache = sorted({c for c in ("__pycache__", ".mypy_cache", ".ipynb_checkpoints",
                                    "wandb", ".venv", "venv")
                        for f in seguidos if f.split("/")[0] == c or f"/{c}/" in f"/{f}"})
        if pesados:
            r.ojo(f"hay {len(pesados)} fichero(s) de pesos versionados: "
                  f"{', '.join(pesados[:3])}. Los pesos no van al repositorio")
        if cache:
            r.ojo(f"hay ficheros de cache versionados: {', '.join(cache)}. "
                  f"Anadelos al .gitignore y quitalos con git rm -r --cached")

    return r.cerrar()


# --------------------------------------------------------------------------
# 2 · dependencias con version fijada
# --------------------------------------------------------------------------

def check_dependencias(raiz: Path) -> Resultado:
    r = Resultado("dependencias", "Dependencias con version fijada", 15, critico=True)
    req = raiz / "requirements.txt"
    if not req.is_file():
        r.fallo("no hay requirements.txt")
        return r.cerrar()

    lineas = [l.strip() for l in lee(req).splitlines()]
    paquetes = [l for l in lineas
                if l and not l.startswith("#") and not l.startswith("-")]
    if not paquetes:
        r.fallo("requirements.txt esta vacio")
        return r.cerrar()

    r.bien(f"{len(paquetes)} dependencias declaradas", 5)

    fijadas = [p for p in paquetes if re.search(r"(==|@)", p)]
    ratio = len(fijadas) / len(paquetes)
    if ratio == 1.0:
        r.bien("todas con version exacta (==)", 10)
    elif ratio >= 0.8:
        r.bien(f"{len(fijadas)}/{len(paquetes)} con version exacta", 6)
        r.fallo("faltan por fijar: " + ", ".join(p for p in paquetes if p not in fijadas))
    else:
        r.fallo(f"solo {len(fijadas)}/{len(paquetes)} tienen version exacta. "
                f"Sin versiones fijadas el entorno no es reproducible")

    if not any(re.match(r"torch\b", p) for p in paquetes):
        r.ojo("no aparece torch en requirements.txt")

    return r.cerrar()


# --------------------------------------------------------------------------
# 3 · seeds
# --------------------------------------------------------------------------

def check_seeds(raiz: Path) -> Resultado:
    r = Resultado("seeds", "Semillas aleatorias fijadas", 12)
    py = ficheros_py(raiz)
    if not py:
        r.fallo("no hay ficheros .py que revisar")
        return r.cerrar()

    texto = "\n".join(lee(p) for p in py)

    familias = {
        "torch": r"torch\.manual_seed\s*\(",
        "numpy": r"(np|numpy)\.random\.(seed|default_rng)\s*\(",
        "random": r"\brandom\.seed\s*\(",
    }
    hallados = [k for k, pat in familias.items() if re.search(pat, texto)]
    if hallados:
        r.bien(f"seeds fijadas para: {', '.join(hallados)}", 4 * len(hallados))
    else:
        r.fallo("no se fija ninguna semilla. Sin seeds los resultados no se reproducen")

    if re.search(r"cuda\.manual_seed", texto):
        r.bien("tambien se fija la semilla de CUDA", 0)

    # que este centralizado, no repartido
    con_seed = [p for p in py if re.search(r"manual_seed|random\.seed|default_rng", lee(p))]
    if len(con_seed) > 2:
        r.ojo(f"las seeds se fijan en {len(con_seed)} ficheros distintos; "
              f"mejor centralizarlas en uno solo")

    return r.cerrar()


# --------------------------------------------------------------------------
# 4 · smoke test (la regla 5/20)
# --------------------------------------------------------------------------

def check_smoke(raiz: Path, ejecutar: bool) -> Resultado:
    r = Resultado("smoke", "Smoke test end-to-end (regla 5/20)", 25, critico=True)
    run = raiz / "run.py"
    if not run.is_file():
        r.fallo("no hay run.py")
        return r.cerrar()

    fuente = lee(run)
    if "--smoke" in fuente or "smoke" in fuente:
        r.bien("run.py contempla el modo --smoke", 5)
    else:
        r.fallo("run.py no acepta --smoke")
        return r.cerrar()

    if not ejecutar:
        r.ojo("ejecucion saltada (--no-smoke); solo se ha comprobado que existe el modo")
        return r.cerrar()

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "run.py", "--smoke"],
            cwd=str(raiz), capture_output=True, text=True,
            timeout=SMOKE_LIMITE_S, errors="replace",
        )
    except subprocess.TimeoutExpired:
        r.fallo(f"el smoke test no termino en {SMOKE_LIMITE_S // 60} minutos. "
                f"Incumple la regla 5/20")
        return r.cerrar()
    except OSError as e:
        r.fallo(f"no se pudo ejecutar run.py: {e}")
        return r.cerrar()

    dur = time.time() - t0
    if proc.returncode != 0:
        cola = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        r.fallo(f"el smoke test fallo con codigo {proc.returncode}")
        for l in cola:
            r.fallo(f"    | {l}")
        return r.cerrar()

    r.bien(f"smoke test en verde en {dur:.0f} s", 12)
    if dur <= SMOKE_LIMITE_S / 2:
        r.bien("holgadamente dentro del limite de 5 minutos", 4)
    else:
        r.ojo(f"tarda {dur:.0f} s de un maximo de {SMOKE_LIMITE_S}; va justo")
        r.pts += 2

    salida = (proc.stdout or "") + (proc.stderr or "")
    if re.search(r"\b\d+\.\d+\b", salida):
        r.bien("la ejecucion imprime al menos una metrica numerica", 4)
    else:
        r.fallo("la ejecucion no imprime ninguna metrica; "
                "un smoke test tiene que demostrar que el pipeline llega al final")

    return r.cerrar()


# --------------------------------------------------------------------------
# 5 · README ejecutable
# --------------------------------------------------------------------------

def check_readme(raiz: Path) -> Resultado:
    r = Resultado("readme", "README con los comandos exactos", 13)
    rd = raiz / "README.md"
    if not rd.is_file():
        r.fallo("no hay README.md")
        return r.cerrar()

    txt = lee(rd)
    if len(txt) < 400:
        r.fallo("el README tiene menos de 400 caracteres; no describe nada")
    else:
        r.bien("README con contenido", 3)

    quiere = {
        "instalacion": (r"pip install|conda (env )?create|environment\.yml", 3,
                        "como instalar las dependencias"),
        "datos": (r"get_data|download|descarga|dataset", 2,
                  "como conseguir los datos"),
        "ejecucion": (r"python run\.py", 3,
                      "el comando exacto de ejecucion"),
        "paper": (r"arxiv|doi|https?://", 2,
                  "enlace al paper o referencia tecnica"),
    }
    for _, (pat, pts, desc) in quiere.items():
        if re.search(pat, txt, re.I):
            r.bien(f"documenta {desc}", pts)
        else:
            r.fallo(f"el README no documenta {desc}")

    if "```" not in txt:
        r.ojo("no hay ningun bloque de codigo; los comandos deberian ir en bloques")

    if re.search(r"[A-Z]:\\\\|[A-Z]:/Users/|/home/[a-z]+/", txt):
        r.fallo("hay rutas absolutas en el README; nadie mas las puede usar")

    return r.cerrar()


# --------------------------------------------------------------------------
# 6 · bitacora de uso de IA
# --------------------------------------------------------------------------

def check_ai_usage(raiz: Path) -> Resultado:
    r = Resultado("ai_usage", "Bitacora de uso de IA", 10)
    f = raiz / "AI_USAGE.md"
    if not f.is_file():
        r.fallo("no hay AI_USAGE.md. Es obligatorio segun la guia docente, "
                "aunque no hayais usado ninguna herramienta (en ese caso, decidlo ahi)")
        return r.cerrar()

    txt = lee(f)
    r.bien("AI_USAGE.md presente", 3)

    # los ejemplos de la plantilla van dentro de comentarios HTML: no cuentan
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.S)

    filas = [l for l in txt.splitlines()
             if l.strip().startswith("|") and l.count("|") >= 4]
    # descontar cabecera y separador de la tabla plantilla
    filas_datos = [l for l in filas if not re.match(r"^\s*\|[\s\-:|]+\|\s*$", l)]
    filas_datos = filas_datos[1:] if filas_datos else []

    if len(filas_datos) >= 4:
        r.bien(f"{len(filas_datos)} entradas registradas", 4)
    elif len(filas_datos) >= 1:
        r.bien(f"{len(filas_datos)} entradas registradas", 2)
        r.ojo("pocas entradas para un proyecto de tres meses")
    else:
        r.fallo("el fichero existe pero no tiene entradas")

    # buscar el criterio SOLO en las filas de la tabla, no en las instrucciones
    cuerpo = "\n".join(filas_datos)
    if re.search(r"rechaz|descart|corregi|modificam|no lo us|lo cambiamos", cuerpo, re.I):
        r.bien("hay al menos una decision critica documentada (algo rechazado o corregido)", 3)
    else:
        r.ojo("no consta ninguna sugerencia rechazada o corregida. "
              "La rubrica valora el criterio, no la abstinencia")

    return r.cerrar()


# --------------------------------------------------------------------------
# 7 · seguimiento de experimentos
# --------------------------------------------------------------------------

def check_tracking(raiz: Path) -> Resultado:
    r = Resultado("tracking", "Seguimiento de experimentos", 10)
    py = ficheros_py(raiz)
    codigo = "\n".join(lee(p) for p in py)
    docs = lee(raiz / "README.md")

    if re.search(r"\bimport wandb\b|\bwandb\.init\s*\(", codigo):
        r.bien("el codigo registra experimentos en W&B", 5)
    elif re.search(r"mlflow|tensorboard", codigo, re.I):
        r.bien("usa otra herramienta de seguimiento (MLflow / TensorBoard)", 4)
    else:
        r.fallo("no se registra ningun experimento. Se exige W&B (o equivalente justificado)")

    m = re.search(r"https?://(wandb\.ai|api\.wandb\.ai)/\S+", docs)
    if m:
        r.bien("el README enlaza el proyecto de W&B", 5)
    else:
        r.fallo("el README no enlaza el proyecto de seguimiento. "
                "Sin enlace publico no se pueden ver los runs")

    return r.cerrar()


# --------------------------------------------------------------------------
# informe
# --------------------------------------------------------------------------

def informe(res: list[Resultado], raiz: Path) -> tuple[int, bool]:
    total = sum(x.pts for x in res)
    maximo = sum(x.max_pts for x in res)
    nota = round(100 * total / maximo) if maximo else 0
    criticos_a_cero = [x for x in res if x.critico and x.pts == 0]
    pasa = nota >= APROBADO and not criticos_a_cero

    ancho = 74
    print()
    print("=" * ancho)
    print(f" tiiac-check  ·  {raiz.resolve().name}")
    print("=" * ancho)

    for x in res:
        estado = "OK  " if x.pts == x.max_pts else ("--  " if x.pts else "FAIL")
        crit = "  [critico]" if x.critico else ""
        print(f"\n[{estado}] {x.titulo}   {x.pts}/{x.max_pts}{crit}")
        for m in x.ok:
            print(f"        + {m}")
        for m in x.mal:
            print(f"        X {m}")
        for m in x.aviso:
            print(f"        ! {m}")

    print()
    print("-" * ancho)
    print(f" PUNTUACION: {nota}/100     (aprueba con {APROBADO})")
    if criticos_a_cero:
        print(f" BLOQUEADO por bloques criticos a cero: "
              f"{', '.join(x.clave for x in criticos_a_cero)}")
    print(f" RESULTADO:  {'VERDE · la entrega se corrige' if pasa else 'ROJO · hay que subsanar'}")
    print("-" * ancho)
    if not pasa:
        print(" Ventana de subsanacion: 72 h con penalizacion. Arregla las lineas con X.")
    print()
    return nota, pasa


def main() -> int:
    ap = argparse.ArgumentParser(description="Comprobador de reproducibilidad de TIIAC")
    ap.add_argument("ruta", nargs="?", default=".", help="directorio del repositorio")
    ap.add_argument("--no-smoke", action="store_true",
                    help="no ejecuta run.py --smoke (mas rapido, menos fiable)")
    ap.add_argument("--json", action="store_true", help="salida JSON para CI")
    args = ap.parse_args()

    raiz = Path(args.ruta).resolve()
    if not raiz.is_dir():
        print(f"error: {raiz} no es un directorio", file=sys.stderr)
        return 2

    res = [
        check_estructura(raiz),
        check_dependencias(raiz),
        check_seeds(raiz),
        check_smoke(raiz, ejecutar=not args.no_smoke),
        check_readme(raiz),
        check_ai_usage(raiz),
        check_tracking(raiz),
    ]

    if args.json:
        total = sum(x.pts for x in res)
        maximo = sum(x.max_pts for x in res)
        nota = round(100 * total / maximo) if maximo else 0
        pasa = nota >= APROBADO and not [x for x in res if x.critico and x.pts == 0]
        print(json.dumps({
            "score": nota,
            "pass": pasa,
            "checks": [{"key": x.clave, "title": x.titulo, "points": x.pts,
                        "max": x.max_pts, "critical": x.critico,
                        "ok": x.ok, "fail": x.mal, "warn": x.aviso} for x in res],
        }, ensure_ascii=False, indent=2))
        return 0 if pasa else 1

    _, pasa = informe(res, raiz)
    return 0 if pasa else 1


if __name__ == "__main__":
    sys.exit(main())
