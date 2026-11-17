"""El aplicativo: sirve la interfaz y la API que la alimenta.

    python -m src.app.serve                 # http://127.0.0.1:8501
    python -m src.app.serve --puerto 8000 --host 0.0.0.0

La pagina (src/app/static/) es la interfaz disenada en Claude Design, portada
tal cual: un diagrama 3D de la red con zoom a los canales y a las neuronas,
los parametros del articulo en secciones plegables, la optimizacion paso a
paso y sus metricas en tiempo real. Todo eso vive en el navegador; aqui solo
hay rutas que llaman a servicio.py, que es quien habla con la red.

No usa Streamlit: la interfaz guarda mucho estado propio (la camara, el nivel
de zoom, el cursor de las graficas) y recibe el progreso de una optimizacion
que dura minutos, y el modelo de "re-ejecutar el script en cada clic" no
encaja. Tampoco hace falta ningun framework: con la biblioteca estandar basta.

Carga un checkpoint y responde. NO entrena ni importa nada de src/training/.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.app.servicio import Servicio  # noqa: E402

ESTATICOS = Path(__file__).resolve().parent / "static"
_ID = r"([0-9a-f]{6,32})"
_ARQ = r"([a-z0-9]+)"


def crear_manejador(servicio: Servicio) -> type[BaseHTTPRequestHandler]:
    """Una clase de manejador HTTP atada a ese servicio.

    Rutas (todas devuelven JSON salvo las imagenes):
        GET  /                                     la pagina
        GET  /static/<fichero>                     js, css y librerias
        GET  /api/info                             modelos, mandos, muestras, especies
        POST /api/fotos                            sube una foto (el cuerpo son los bytes)
        POST /api/fotos/muestra                    {"nombre"}: una de data/muestras
        POST /api/fotos/inicial                    la foto de arranque
        GET  /api/fotos/<id>/foto.png              la foto como la ve el modelo
        GET  /api/fotos/<id>/rgb.png               sus canales R, G y B
        GET  /api/fotos/<id>/<arq>/analisis        datos del diagrama
        GET  /api/fotos/<id>/<arq>/atlas/<e>.png   mapas de todos los canales de una etapa
        GET  /api/fotos/<id>/<arq>/mapa?capa=&canal=   la foto con ese mapa superpuesto
        POST /api/trabajos                         lanza una optimizacion
        GET  /api/trabajos/<id>?desde=&capturas=&vista=  su progreso
    """

    class Manejador(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, formato: str, *args) -> None:  # noqa: D401 - silencia el log por peticion
            return

        # -------------------------------------------------------- respuestas
        def _enviar(self, cuerpo: bytes, tipo: str, estado: int = 200, cache: bool = False) -> None:
            self.send_response(estado)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def _json(self, datos, estado: int = 200) -> None:
            self._enviar(json.dumps(datos, ensure_ascii=False).encode(), "application/json; charset=utf-8", estado)

        def _error(self, estado: int, mensaje: str) -> None:
            self._json({"error": mensaje}, estado)

        def _cuerpo(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _estatico(self, ruta: str) -> None:
            fichero = (ESTATICOS / ruta).resolve()
            if ESTATICOS not in fichero.parents or not fichero.is_file():
                self._error(404, "no existe")
                return
            tipo = mimetypes.guess_type(fichero.name)[0] or "application/octet-stream"
            if fichero.suffix in (".js", ".mjs"):
                tipo = "text/javascript; charset=utf-8"
            self._enviar(fichero.read_bytes(), tipo)

        # -------------------------------------------------------- rutas
        def _despachar(self, metodo: str) -> None:
            url = urlparse(self.path)
            ruta, q = unquote(url.path), parse_qs(url.query)
            arg = lambda k, d=None: q.get(k, [d])[0]  # noqa: E731
            try:
                if metodo == "GET":
                    if ruta in ("/", "/index.html"):
                        return self._estatico("index.html")
                    if ruta.startswith("/static/"):
                        return self._estatico(ruta[len("/static/"):])
                    if ruta == "/api/info":
                        return self._json(servicio.info())
                    if m := re.fullmatch(rf"/api/fotos/{_ID}/foto\.png", ruta):
                        return self._enviar(servicio.foto_png(m[1]), "image/png", cache=True)
                    if m := re.fullmatch(rf"/api/fotos/{_ID}/rgb\.png", ruta):
                        return self._enviar(servicio.canales_png(m[1]), "image/png", cache=True)
                    if m := re.fullmatch(rf"/api/fotos/{_ID}/{_ARQ}/analisis", ruta):
                        return self._json(servicio.analizar(m[1], m[2]))
                    if m := re.fullmatch(rf"/api/fotos/{_ID}/{_ARQ}/atlas/([0-3])\.png", ruta):
                        return self._enviar(servicio.atlas_png(m[1], m[2], int(m[3])), "image/png", cache=True)
                    if m := re.fullmatch(rf"/api/fotos/{_ID}/{_ARQ}/mapa", ruta):
                        return self._json(servicio.mapa(m[1], m[2], arg("capa", ""), int(arg("canal", "0"))))
                    if m := re.fullmatch(rf"/api/trabajos/{_ID}", ruta):
                        return self._json(servicio.estado(m[1], int(arg("desde", "0")),
                                                          int(arg("capturas", "0")), int(arg("vista", "0"))))
                elif metodo == "POST":
                    if ruta == "/api/fotos":
                        nombre = unquote(self.headers.get("X-Nombre") or "foto")
                        return self._json(servicio.anadir_foto(self._cuerpo(), nombre))
                    if ruta == "/api/fotos/muestra":
                        return self._json(servicio.anadir_muestra(json.loads(self._cuerpo())["nombre"]))
                    if ruta == "/api/fotos/inicial":
                        return self._json(servicio.anadir_inicial())
                    if ruta == "/api/trabajos":
                        return self._json({"id": servicio.lanzar(json.loads(self._cuerpo()))})
                self._error(404, f"no hay ninguna ruta {metodo} {ruta}")
            except KeyError as e:
                self._error(404, str(e).strip("'\""))
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                self._error(400, str(e))
            except Exception as e:  # noqa: BLE001 - un fallo inesperado no debe tumbar el servidor
                traceback.print_exc()
                self._error(500, f"{type(e).__name__}: {e}")

        def do_GET(self) -> None:  # noqa: N802 - nombre impuesto por BaseHTTPRequestHandler
            self._despachar("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._despachar("POST")

    return Manejador


def crear_servidor(servicio: Servicio, host: str, puerto: int) -> ThreadingHTTPServer:
    """El servidor listo para serve_forever(). Con puerto 0 elige uno libre."""
    servidor = ThreadingHTTPServer((host, puerto), crear_manejador(servicio))
    servidor.daemon_threads = True
    return servidor


def main(argv: list[str] | None = None) -> int:
    """Arranca el aplicativo con configs/app.yaml."""
    servicio = Servicio()
    cfg = servicio.app["servidor"]
    p = argparse.ArgumentParser(description="Interfaz de visualizacion de caracteristicas")
    p.add_argument("--host", default=cfg["host"])
    p.add_argument("--puerto", type=int, default=int(cfg["puerto"]))
    args = p.parse_args(argv)

    servidor = crear_servidor(servicio, args.host, args.puerto)
    print(f"Visualizacion de caracteristicas en http://{args.host}:{servidor.server_port}  (Ctrl+C para salir)")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
