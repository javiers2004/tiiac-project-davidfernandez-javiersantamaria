"""Objetivos de optimizacion: que parte de la red queremos maximizar.

Replica los objetivos del articulo de Olah et al., "Feature Visualization"
(Distill, 2017), seccion "Optimization Objectives":

    Neuron    layer_n[x, y, z]     una posicion concreta de un canal
    Channel   layer_n[:, :, z]     el canal entero, promediado en el espacio
    Layer     layer_n[:, :, :]^2   la capa entera al cuadrado (DeepDream)
    Logit     pre_softmax[k]       la salida de la clase k antes de la softmax
    Prob.     softmax[k]           la probabilidad de la clase k

Y de la seccion "The Interaction between Neurons":

    Direction  v . layer_n[x, y, :]  una direccion cualquiera del espacio de
                                     activaciones, promediada en el espacio
    Suma       A + B + ...           varios objetivos a la vez (aritmetica de
                                     neuronas, "jointly optimized")
    Interp.    (1-w_i) A + w_i B     una fila de imagenes que va de A a B

La unica diferencia entre ellos es la reduccion: que parte de la capa se toma
y como se resume en un escalar. Todo lo demas (parametrizacion en Fourier,
decorrelacion de color, robustez) es comun y vive en feature_viz.py.

ORDEN DE EJES. El articulo y lucid trabajan en NHWC y escriben layer[x, y, z].
PyTorch es NCHW. La traduccion es:

    lucid      layer[:, x, y, z]      (N, H, W, C)
    PyTorch    act[:, z, x, y]        (N, C, H, W)

Ojo a un detalle de lucid que se presta a error: su `x` indexa shape[1], que en
NHWC es el ALTO. Aqui se respeta esa convencion, asi que `x` es el indice de
FILA e `y` el de COLUMNA. Confundirlos no da error ni forma invalida: devuelve
la neurona traspuesta y una imagen que parece correcta.

Referencia: lucid/optvis/objectives.py y objectives_util.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FormaCapa:
    """Lo que una capa ofrece: cuantos canales y que extension espacial.

    La aplicacion la usa para saber que rangos puede ofrecer el usuario sin
    tener que cablear nada: los limites de z y de (x, y) dependen de la capa
    elegida y cambian de 56x56 a 7x7 segun la profundidad.
    """

    capa: str
    canales: int
    alto: int
    ancho: int

    @property
    def es_espacial(self) -> bool:
        """False en capas sin extension espacial, como la lineal de salida."""
        return self.alto > 0 and self.ancho > 0

    @property
    def centro(self) -> tuple[int, int] | None:
        """(x, y) central, el valor por defecto del objetivo Neuron."""
        if not self.es_espacial:
            return None
        return self.alto // 2, self.ancho // 2


@dataclass(frozen=True)
class Canal:
    """Objetivo Channel: layer_n[:, :, z].

    Media del canal z sobre todas las posiciones. Es el objetivo que el
    articulo usa para la mayoria de sus imagenes.

    En una capa sin extension espacial (la lineal de salida) se reduce al
    logit de la clase z, que es justo el objetivo Class Logits del articulo.
    Da el mismo numero que Logit(z); Logit existe para nombrarlo y para
    fallar si la capa no es la de salida.
    """

    z: int

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        _validar_canal(act, self.z)
        return act[:, self.z].mean()

    def __str__(self) -> str:
        return f"canal[{self.z}]"


@dataclass(frozen=True)
class Neurona:
    """Objetivo Neuron: layer_n[x, y, z].

    El canal z en UNA sola posicion. Si x o y son None se usa el centro del
    mapa, con division entera, igual que lucid.

    Aviso para la interfaz: una sola posicion solo recibe gradiente de su
    campo receptivo. En las capas tempranas eso es una fraccion pequena del
    lienzo y el resto se queda en el ruido inicial. `campo_receptivo()` da la
    cifra exacta para avisar al usuario o para ajustar el tamano de imagen.
    """

    z: int
    x: int | None = None
    y: int | None = None

    def posicion(self, alto: int, ancho: int) -> tuple[int, int]:
        """Resuelve (x, y), sustituyendo None por el centro y validando rango."""
        x = alto // 2 if self.x is None else self.x
        y = ancho // 2 if self.y is None else self.y
        if not 0 <= x < alto:
            raise IndexError(f"x={x} fuera del mapa, que tiene {alto} filas")
        if not 0 <= y < ancho:
            raise IndexError(f"y={y} fuera del mapa, que tiene {ancho} columnas")
        return x, y

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        if act.dim() != 4:
            raise ValueError(
                f"el objetivo Neuron necesita una capa con extension espacial "
                f"(N,C,H,W); esta da {tuple(act.shape)}. En la capa de salida "
                f"usa el objetivo 'logit'."
            )
        _validar_canal(act, self.z)
        x, y = self.posicion(act.shape[2], act.shape[3])
        # act[:, z, x, y]: x indexa el alto y y el ancho. Ver cabecera.
        return act[:, self.z, x, y].mean()

    def __str__(self) -> str:
        pos = "centro" if self.x is None and self.y is None else f"{self.x},{self.y}"
        return f"neurona[{self.z}] en {pos}"


@dataclass(frozen=True)
class Capa:
    """Objetivo Layer (DeepDream): layer_n[:, :, :]^2.

    Media del CUADRADO de toda la capa: todos los canales, todas las
    posiciones. No elige unidad, asi que no lleva z: la imagen amplifica lo
    que la capa ya "ve" en ella, sea lo que sea.

    El cuadrado no es un detalle. Con la media a secas, repartir una misma
    cantidad de activacion entre muchas unidades o concentrarla en unas pocas
    da igual; con el cuadrado, concentrar paga mas. Por eso los DeepDream
    acaban dominados por unos pocos patrones repetidos en vez de por un gris
    que excite un poco a todo. Y en una capa pre-ReLU (una conv suelta) el
    cuadrado cuenta tambien las activaciones negativas: empujar una unidad
    hacia abajo puntua igual que hacia arriba.

    Referencia: lucid.optvis.objectives.deepdream, que es reduce_mean(T**2).
    """

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        return act.pow(2).mean()

    def __str__(self) -> str:
        return "capa^2"


@dataclass(frozen=True)
class Logit:
    """Objetivo Class Logits: pre_softmax[k].

    La salida de la clase k ANTES de la softmax, en la capa de salida ('fc').
    Es lo que el articulo usa para visualizar clases: la imagen tiene que
    parecerse a la clase, no solo dejar de parecerse a las demas (ver
    Probabilidad).

    Numericamente es Canal(k) sobre 'fc'. La diferencia es que este falla si
    la capa no es la de salida: con Canal(k) sobre 'layer4' no hay error, se
    optimiza el canal k de una conv y la imagen se etiqueta como una clase.
    """

    clase: int

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        _validar_salida(act, "Class Logits")
        _validar_clase(act, self.clase)
        return act[:, self.clase].mean()

    def __str__(self) -> str:
        return f"logit[{self.clase}]"


@dataclass(frozen=True)
class Probabilidad:
    """Objetivo Class Probability: softmax[k].

    La probabilidad de la clase k. Se aplica la softmax aqui, sobre los
    logits que captura el hook de 'fc', porque el modelo no la tiene como
    modulo propio.

    Esta para replicar la comparacion del articulo, no como objetivo por
    defecto. softmax[k] = exp(z_k) / sum_j exp(z_j): sube igual subiendo z_k
    que hundiendo las otras 1080 clases, y lo segundo suele ser mas facil.
    La imagen resultante tiende a ser "lo que no es ninguna otra clase" mas
    que "lo que es la clase k". Ademas se satura: cuando la probabilidad se
    acerca a 1 el gradiente se apaga, aunque el logit pueda seguir subiendo.
    """

    clase: int

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        _validar_salida(act, "Class Probability")
        _validar_clase(act, self.clase)
        return act.softmax(dim=-1)[:, self.clase].mean()

    def __str__(self) -> str:
        return f"softmax[{self.clase}]"


@dataclass(frozen=True)
class Direccion:
    """Objetivo Direction: v . layer_n[x, y, :], promediado en el espacio.

    El articulo llama espacio de activaciones al de todas las combinaciones de
    canales de una capa. Cada canal es un eje; una combinacion de canales es
    un vector v cualquiera de ese espacio. En cada posicion se hace el
    producto escalar del vector de canales con v, y se promedia. Si v fuese el
    eje z (todo ceros salvo un 1 en z) esto seria exactamente Canal(z).

    v es aleatorio: normal y de norma 1. Se guarda la semilla y no el vector,
    asi el objetivo sigue siendo comparable y serializable como los demas, y
    no necesita saber cuantos canales tiene la capa hasta que se aplica.

    Ojo en capas post-ReLU (las salidas de bloque): los canales no bajan de
    cero, asi que los pesos negativos de v solo pueden pedir que ese canal se
    calle. En una conv pre-ReLU si pueden empujarlo por debajo de cero.

    Referencia: lucid.optvis.objectives.direction con cossim_pow=0.
    """

    semilla: int

    def vector(self, canales: int) -> torch.Tensor:
        """La direccion v para una capa de `canales` canales, de norma 1."""
        # Generador propio: no toca el estado global, que fija seed.py.
        generador = torch.Generator().manual_seed(self.semilla)
        v = torch.randn(canales, generator=generador)
        return v / v.norm()

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        v = self.vector(act.shape[1]).to(act.device)
        # (C,) -> (1, C, 1, 1) en una conv y (1, C) en la lineal de salida.
        v = v.view(1, -1, *([1] * (act.dim() - 2)))
        return (act * v).sum(dim=1).mean()

    def __str__(self) -> str:
        return f"direccion[semilla {self.semilla}]"


@dataclass(frozen=True)
class Suma:
    """Varios objetivos maximizados a la vez.

    Es el "jointly optimized" de la seccion "Interaction between Neurons" del
    articulo: la imagen resultante tiene que contentar a todas las unidades a
    la vez, y eso revela como se combinan. Es aritmetica de neuronas: su
    ejemplo es que una neurona "blanco y negro" mas una "mosaico" da un
    mosaico en blanco y negro.

    Desde el YAML (tipo 'suma') se construye como suma de canales de la
    misma capa; desde codigo admite cualquier mezcla de objetivos.
    """

    partes: tuple

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        if not self.partes:
            raise ValueError("una Suma necesita al menos un objetivo")
        return sum(parte(act) for parte in self.partes)

    def componentes(self, act: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Cada sumando por separado, para comprobar que ninguno esta muerto.

        Si uno lo esta, la suma sigue teniendo gradiente por los demas y el
        fallo no se veria: la imagen ignoraria esa neurona sin avisar.
        """
        return [(str(parte), parte(act)) for parte in self.partes]

    def __str__(self) -> str:
        return " + ".join(str(parte) for parte in self.partes)


@dataclass(frozen=True)
class Interpolacion:
    """Interpolacion entre dos canales de la misma capa, a lo largo del lote.

    La imagen i de una fila de N maximiza (1 - w_i) * Canal(a) + w_i * Canal(b)
    con w_i = i / (N - 1): la primera es solo a, la ultima solo b, y las de
    en medio muestran como se combinan. Es la "linear interpolation between
    the individual channel objectives" del articulo.

    Necesita imagen.lote >= 2. Sola no basta para que la fila se vea como una
    transicion: el articulo le anade la alineacion (feature_viz.alineacion) y
    una parametrizacion compartida entre las imagenes.

    Se promedia sobre el lote, como el resto de objetivos, en vez de sumar como
    lucid: asi la activacion que se registra es comparable a la de Canal. Con
    Adam la escala no cambia la optimizacion, pero si el peso relativo de las
    penalizaciones; los pesos del YAML estan pensados para esta convencion.

    Referencia: lucid.optvis.objectives.channel_interpolate.
    """

    a: int
    b: int

    @staticmethod
    def pesos(n: int) -> torch.Tensor:
        """w_i de cada imagen de una fila de n: de 0 (solo a) a 1 (solo b)."""
        return torch.linspace(0.0, 1.0, n)

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        n = act.shape[0]
        if n < 2:
            raise ValueError(
                f"la interpolacion necesita al menos 2 imagenes y el lote tiene "
                f"{n}. Sube imagen.lote en el YAML (p.ej. --lote 6)."
            )
        _validar_canal(act, self.a)
        _validar_canal(act, self.b)
        w = self.pesos(n).to(act.device)
        # Media espacial de cada canal, imagen a imagen: (N,).
        a = act[:, self.a].reshape(n, -1).mean(dim=1)
        b = act[:, self.b].reshape(n, -1).mean(dim=1)
        return ((1 - w) * a + w * b).mean()

    def componentes(self, act: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """a en la primera imagen y b en la ultima, por separado.

        La primera imagen solo puntua por a. Si a esta muerto, b sigue dando
        gradiente al objetivo entero y nadie avisaria de que esa imagen no
        recibe nada de el.
        """
        return [(f"canal[{self.a}] en la primera imagen", act[:1, self.a].mean()),
                (f"canal[{self.b}] en la ultima imagen", act[-1:, self.b].mean())]

    def __str__(self) -> str:
        return f"interpolacion[{self.a} -> {self.b}]"


OBJETIVOS = {"canal": Canal, "neurona": Neurona, "capa": Capa,
             "logit": Logit, "probabilidad": Probabilidad,
             "direccion": Direccion, "suma": Suma,
             "interpolacion": Interpolacion}

Objetivo = (Canal | Neurona | Capa | Logit | Probabilidad | Direccion | Suma
            | Interpolacion)


def construir_objetivo(ocfg: dict) -> Objetivo:
    """Crea el objetivo a partir de la subconfiguracion 'objetivo' del YAML.

    Args:
        ocfg: dict con 'tipo' (una clave de OBJETIVOS), 'canal' (el indice
            z, que en 'logit' y 'probabilidad' es la clase y que 'capa' y
            'direccion' no usan), 'direccion' (la semilla del vector, solo
            para 'direccion'), 'canales' (la lista de canales en 'suma';
            el de partida y el de llegada en 'interpolacion') y, solo para
            'neurona', 'x' e 'y' opcionales.

    Returns:
        El objetivo, invocable sobre el tensor de activacion de la capa.
    """
    tipo = str(ocfg.get("tipo", "canal"))
    if tipo not in OBJETIVOS:
        raise ValueError(f"objetivo desconocido: '{tipo}'. "
                         f"Opciones: {', '.join(OBJETIVOS)}")
    if tipo == "capa":
        return Capa()
    if tipo == "direccion":
        return Direccion(int(ocfg["direccion"]))
    if tipo == "suma":
        return Suma(tuple(Canal(int(z)) for z in ocfg["canales"]))
    if tipo == "interpolacion":
        canales = list(ocfg["canales"])
        if len(canales) != 2:
            raise ValueError(f"la interpolacion va de un canal a otro: 'canales' "
                             f"tiene que tener 2 y tiene {len(canales)}")
        return Interpolacion(int(canales[0]), int(canales[1]))
    z = int(ocfg["canal"])
    if tipo == "canal":
        return Canal(z)
    if tipo == "logit":
        return Logit(z)
    if tipo == "probabilidad":
        return Probabilidad(z)
    x, y = ocfg.get("x"), ocfg.get("y")
    return Neurona(z, None if x is None else int(x), None if y is None else int(y))


def resolver_modulo(modelo: nn.Module, ruta: str) -> nn.Module:
    """Devuelve el submodulo indicado por una ruta con puntos ('layer3.1.conv2')."""
    modulo: nn.Module = modelo
    for parte in ruta.split("."):
        modulo = modulo[int(parte)] if parte.isdigit() else getattr(modulo, parte)
    return modulo


def inspeccionar_capa(modelo: nn.Module, capa: str, tamano: int = 224) -> FormaCapa:
    """Pasa una entrada en vacio para averiguar que forma produce la capa.

    Es lo que la aplicacion necesita para poblar los selectores: cuantos
    canales hay y hasta donde llegan x e y. Depende del tamano de entrada,
    porque los mapas se encogen con el.
    """
    capturado: dict[str, torch.Tensor] = {}
    asa = resolver_modulo(modelo, capa).register_forward_hook(
        lambda _m, _e, salida: capturado.__setitem__("act", salida)
    )
    try:
        with torch.no_grad():
            modelo(torch.zeros(1, 3, tamano, tamano))
    finally:
        asa.remove()

    act = capturado["act"]
    if act.dim() == 4:
        return FormaCapa(capa, act.shape[1], act.shape[2], act.shape[3])
    return FormaCapa(capa, act.shape[1], 0, 0)


def campo_receptivo(modelo: nn.Module, capa: str, tamano: int = 224,
                    repeticiones: int = 4) -> tuple[int, int]:
    """Mide el campo receptivo efectivo de una posicion, en pixeles de entrada.

    Se propaga el gradiente desde una sola posicion del mapa hasta la entrada y
    se mide la caja de pixeles con gradiente no nulo. Se usan entradas
    aleatorias y se acumula la union: con una entrada de ceros los ReLU cortan
    el gradiente y el resultado sale vacio.

    Returns:
        (alto, ancho) en pixeles. (0, 0) si la capa no tiene extension espacial.
    """
    forma = inspeccionar_capa(modelo, capa, tamano)
    if not forma.es_espacial:
        return 0, 0

    capturado: dict[str, torch.Tensor] = {}
    asa = resolver_modulo(modelo, capa).register_forward_hook(
        lambda _m, _e, salida: capturado.__setitem__("act", salida)
    )
    mascara: torch.Tensor | None = None
    try:
        for semilla in range(repeticiones):
            g = torch.Generator().manual_seed(semilla)
            x = torch.randn(1, 3, tamano, tamano, generator=g).requires_grad_(True)
            modelo(x)
            act = capturado["act"]
            # La suma sobre todos los canales: mide la posicion, no un canal
            # concreto, que podria estar muerto y no dar gradiente.
            act[0, :, act.shape[2] // 2, act.shape[3] // 2].sum().backward()
            presente = x.grad.abs().sum(dim=(0, 1)) > 0
            mascara = presente if mascara is None else (mascara | presente)
    finally:
        asa.remove()

    filas = torch.nonzero(mascara.sum(dim=1)).flatten()
    cols = torch.nonzero(mascara.sum(dim=0)).flatten()
    if not len(filas) or not len(cols):
        return 0, 0
    return int(filas[-1] - filas[0] + 1), int(cols[-1] - cols[0] + 1)


def _validar_canal(act: torch.Tensor, z: int) -> None:
    if not 0 <= z < act.shape[1]:
        raise IndexError(f"la capa tiene {act.shape[1]} canales; "
                         f"el canal {z} no existe")


def _validar_salida(act: torch.Tensor, nombre: str) -> None:
    """Los objetivos de clase solo tienen sentido sobre la capa de salida."""
    if act.dim() != 2:
        raise ValueError(
            f"el objetivo {nombre} necesita la capa de salida (N, clases); "
            f"esta da {tuple(act.shape)}. Usa capa: fc."
        )


def _validar_clase(act: torch.Tensor, k: int) -> None:
    if not 0 <= k < act.shape[1]:
        raise IndexError(f"la salida tiene {act.shape[1]} clases; "
                         f"la clase {k} no existe")
