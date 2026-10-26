"""Tests de la visualizacion por maximizacion de activacion.

Lo que de verdad hay que blindar aqui es que se optimiza LA IMAGEN y no los
pesos. Si los pesos se mueven, la imagen resultante no dice nada del modelo
entrenado: dice algo de un modelo que acabamos de estropear, y el fallo es
silencioso.

No usan los .tar: los pesos estan en .gitignore y en CI no existen. Un
resnet18 sin entrenar basta para comprobar la mecanica.
"""

import copy

import pytest
import torch
from torchvision.models import resnet18

from src.interpretability.feature_viz import (
    ImagenFourier,
    UnidadMuerta,
    alineacion,
    aplicar_robustez,
    diversidad,
    maximizar_activacion,
    nombre_salida,
)
from src.interpretability.objetivos import resolver_modulo
from src.utils.seed import fijar_seed

N_CLASES = 10
TAM = 64

CFG = {
    "experimento": {"nombre": "test", "seed": 0},
    "objetivo": {"tipo": "canal", "capa": "fc", "canal": 3},
    "imagen": {"tamano": TAM, "decay_power": 1.0, "sd_inicial": 0.01},
    "optimizacion": {"learning_rate": 0.05, "pasos": 4},
    "regularizacion": {"tv": 0.0, "l2": 0.0},
    "robustez": {"pad": 4, "jitter": 4, "jitter_final": 2,
                 "escalas": [0.95, 1.0, 1.05], "rotaciones": [-5, 0, 5]},
    "normalizacion": {"media": [0.485, 0.456, 0.406],
                      "desviacion": [0.229, 0.224, 0.225]},
}


def _modelo_congelado():
    """Un resnet18 sin entrenar, pero SIEMPRE el mismo.

    Sin fijar la seed, torch siembra su generador global con entropia del
    sistema al arrancar el proceso, asi que cada ejecucion de pytest probaria
    un modelo distinto y los tests serian flaky sin motivo aparente.
    """
    fijar_seed(0)
    modelo = resnet18(num_classes=N_CLASES)
    modelo.eval()
    modelo.requires_grad_(False)
    return modelo


def test_la_imagen_sale_en_el_rango_valido():
    """La sigmoide final tiene que dejar la imagen en [0,1]: si se sale, lo
    que se guarda en el PNG esta recortado y no es lo que se optimizo."""
    imagen = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0))()
    assert imagen.shape == (1, 3, TAM, TAM)
    assert imagen.min() >= 0.0 and imagen.max() <= 1.0


def test_un_lote_da_varias_imagenes_distintas():
    """Cada imagen del lote parte de su propio ruido. Si compartieran el
    ruido, una interpolacion empezaria con todas iguales."""
    imagenes = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0), lote=3)()
    assert imagenes.shape == (3, 3, TAM, TAM)
    assert not torch.allclose(imagenes[0], imagenes[1])


def test_lote_uno_es_la_imagen_de_siempre():
    """Admitir lotes no puede cambiar las visualizaciones que ya existian."""
    a = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0))()
    b = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0), lote=1)()
    assert torch.equal(a, b)


RECETA = [[1, 2], [1, 4], [2, 8]]


def test_la_parte_compartida_de_un_fotograma_es_igual_en_toda_la_fila():
    imagen = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0),
                           lote=4, compartida=[[1, 4]])
    with torch.no_grad():
        imagen.espectro.zero_()
        imagen.compartida[0].normal_(generator=torch.Generator().manual_seed(1))
        fila = imagen()
    for i in range(1, 4):
        assert torch.allclose(fila[0], fila[i])


def test_la_parte_compartida_de_dos_fotogramas_cambia_a_lo_largo_de_la_fila():
    """Con dos fotogramas clave, las imagenes de en medio estan entre las de
    los extremos: la vecina se parece mas que la del otro extremo."""
    imagen = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0),
                           lote=5, compartida=[[2, 4]])
    with torch.no_grad():
        imagen.espectro.zero_()
        imagen.compartida[0].normal_(generator=torch.Generator().manual_seed(1))
        fila = imagen()
    cerca = (fila[0] - fila[1]).abs().mean()
    lejos = (fila[0] - fila[4]).abs().mean()
    assert cerca < lejos


def test_la_parte_compartida_no_cambia_las_imagenes_sueltas():
    """Es una pieza de la interpolacion: con lote=1 ni se crea, asi que todas
    las visualizaciones de una imagen siguen saliendo igual que antes."""
    a = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0))()
    b = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0),
                      compartida=RECETA)()
    assert torch.equal(a, b)


def test_la_parte_compartida_no_altera_el_ruido_propio():
    """Se crea despues del espectro: el ruido de cada imagen es el mismo con
    y sin parte compartida."""
    a = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0), lote=3)
    b = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0), lote=3,
                      compartida=RECETA)
    assert torch.equal(a.espectro, b.espectro)


def test_la_parte_compartida_recibe_gradiente():
    """Si no lo recibiera, se quedaria en el ruido inicial y la fila no
    compartiria nada."""
    imagen = ImagenFourier(TAM, TAM, generador=torch.Generator().manual_seed(0),
                           lote=3, compartida=RECETA)
    imagen()[:, 0].mean().backward()
    assert len(imagen.compartida) == len(RECETA)
    assert all(torch.any(baja.grad != 0) for baja in imagen.compartida)


def test_el_bucle_usa_la_parte_compartida():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "interpolacion", "capa": "layer3", "canales": [1, 2]}
    cfg["imagen"]["lote"] = 3
    cfg["imagen"]["compartida"] = RECETA
    con, _ = maximizar_activacion(cfg, modelo=_modelo_congelado())
    cfg["imagen"]["compartida"] = []
    sin, _ = maximizar_activacion(cfg, modelo=_modelo_congelado())
    assert con.shape == (3, 3, TAM, TAM)
    assert not torch.allclose(con, sin)


def test_la_robustez_conserva_el_tamano():
    """Las transformaciones recortan al tamano de partida. Si no, el modelo
    recibe una entrada de otro tamano y la comparacion entre pasos se rompe."""
    x = torch.rand(1, 3, TAM, TAM)
    g = torch.Generator().manual_seed(0)
    assert aplicar_robustez(x, CFG["robustez"], g).shape == x.shape


def test_la_robustez_transforma_de_verdad():
    """Con la misma imagen, dos pasadas tienen que diferir: si no, no hay
    aleatoriedad y la robustez no esta haciendo nada."""
    x = torch.rand(1, 3, TAM, TAM)
    g = torch.Generator().manual_seed(0)
    a = aplicar_robustez(x, CFG["robustez"], g)
    b = aplicar_robustez(x, CFG["robustez"], g)
    assert not torch.allclose(a, b)


def test_resolver_modulo_navega_las_rutas_con_indices():
    modelo = _modelo_congelado()
    assert resolver_modulo(modelo, "fc") is modelo.fc
    assert resolver_modulo(modelo, "layer3.1.conv2") is modelo.layer3[1].conv2


def test_la_robustez_transforma_igual_todo_el_lote():
    """Si cada imagen recibiera un encuadre distinto, las de una
    interpolacion se desalinearian entre si."""
    x = torch.rand(1, 3, TAM, TAM).repeat(2, 1, 1, 1)
    salida = aplicar_robustez(x, CFG["robustez"], torch.Generator().manual_seed(0))
    assert torch.equal(salida[0], salida[1])


def test_el_bucle_devuelve_el_lote_entero():
    cfg = copy.deepcopy(CFG)
    cfg["imagen"]["lote"] = 3
    imagenes, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())
    assert imagenes.shape == (3, 3, TAM, TAM)
    assert len(historial) == CFG["optimizacion"]["pasos"]


def test_los_pesos_no_se_mueven():
    """El test que importa: optimizamos la imagen, no el modelo."""
    modelo = _modelo_congelado()
    antes = copy.deepcopy(modelo.state_dict())

    maximizar_activacion(CFG, modelo=modelo)

    despues = modelo.state_dict()
    movidos = [k for k in antes if not torch.equal(antes[k], despues[k])]
    assert not movidos, f"estos pesos han cambiado: {movidos}"


def test_la_activacion_sube():
    """Si la activacion no sube, el gradiente no esta llegando a la imagen.

    Se comparan medias y no los extremos: cada paso mide la activacion bajo
    una transformacion aleatoria distinta, asi que un paso suelto puede bajar
    sin que la optimizacion vaya mal. Comparar historial[-1] con historial[0]
    hace el test flaky.
    """
    cfg = copy.deepcopy(CFG)
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    assert len(historial) == 12
    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_deepdream_sube_la_energia_de_la_capa():
    """El mismo bucle, con el objetivo Layer sobre una capa intermedia."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "capa", "capa": "layer2"}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_la_direccion_sube():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "direccion", "capa": "layer3.1.conv2", "direccion": 0}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_el_logit_de_la_clase_sube():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "logit", "capa": "fc", "canal": 3}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:+.4f} a {ultimos:+.4f}"


def test_la_probabilidad_de_la_clase_sube():
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "probabilidad", "capa": "fc", "canal": 3}
    cfg["optimizacion"]["pasos"] = 12
    _, historial = maximizar_activacion(cfg, modelo=_modelo_congelado())

    assert all(0.0 <= p <= 1.0 for p in historial)
    primeros = sum(historial[:4]) / 4
    ultimos = sum(historial[-4:]) / 4
    assert ultimos > primeros, f"de {primeros:.4f} a {ultimos:.4f}"


def test_logit_y_canal_en_fc_dan_la_misma_imagen():
    """Class Logits es Channel sobre fc; la imagen tiene que ser identica."""
    modelo = _modelo_congelado()
    logit_cfg = copy.deepcopy(CFG)
    logit_cfg["objetivo"]["tipo"] = "logit"
    a, _ = maximizar_activacion(CFG, modelo=modelo)
    b, _ = maximizar_activacion(logit_cfg, modelo=modelo)
    assert torch.allclose(a, b)


def test_un_canal_inexistente_falla_pronto():
    """Sin la comprobacion, un indice fuera de rango revienta con un error de
    indexado de torch mucho mas abajo y cuesta de leer."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"]["canal"] = N_CLASES + 5
    with pytest.raises(IndexError, match="no existe"):
        maximizar_activacion(cfg, modelo=_modelo_congelado())


def test_una_unidad_muerta_se_detecta_en_el_primer_paso():
    """Una unidad que un ReLU deja en cero no propaga gradiente: el bucle
    devolveria el ruido inicial sin quejarse. Mas de la mitad de las
    posiciones centrales de layer4 estan asi al arrancar, asi que esto no es
    un caso raro."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "neurona", "capa": "layer4", "canal": 0}

    modelo = _modelo_congelado()
    # s * 0, no zeros_like: una unidad muerta de verdad SIGUE en el grafo, lo
    # que pasa es que su gradiente es cero. Con zeros_like el tensor queda
    # desconectado y backward falla por otro motivo, que no es el que se
    # quiere probar.
    modelo.layer4.register_forward_hook(lambda _m, _e, s: s * 0)

    with pytest.raises(UnidadMuerta, match="no propaga gradiente"):
        maximizar_activacion(cfg, modelo=modelo)


# --- Alineacion -----------------------------------------------------------

def test_la_alineacion_es_cero_si_el_lote_es_igual():
    act = torch.randn(1, 4, 5, 5).repeat(3, 1, 1, 1)
    assert alineacion(act, 2.0, 4).item() == 0.0


def test_la_alineacion_de_dos_imagenes():
    a, b = torch.zeros(1, 2, 3, 3), torch.ones(1, 2, 3, 3)
    assert alineacion(torch.cat([a, b]), 2.0, 4).item() == pytest.approx(0.5)


def test_las_parejas_lejanas_pesan_menos():
    """Con tres imagenes 0, 0, 1: la pareja (1,2) pesa 1/2 y la (0,2), que
    estan a distancia 2, pesa 1/4. Si el decaimiento no se aplicase, o se
    aplicase al reves, el total seria otro."""
    act = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1, 1)
    assert alineacion(act, 2.0, 4).item() == pytest.approx(0.5 + 0.25)
    assert alineacion(act, 2.0, 1).item() == pytest.approx(0.5)


def test_la_alineacion_acerca_las_imagenes_del_lote():
    """El efecto que se busca: con alineacion fuerte, las imagenes de la fila
    acaban mas parecidas entre si que sin ella."""
    def separacion(peso):
        cfg = copy.deepcopy(CFG)
        cfg["imagen"]["lote"] = 3
        cfg["optimizacion"]["pasos"] = 12
        cfg["regularizacion"]["alineacion"] = {"peso": peso, "capas": ["layer1"],
                                               "decaimiento": 2.0, "alcance": 4}
        imagenes, _ = maximizar_activacion(cfg, modelo=_modelo_congelado())
        return (imagenes[1:] - imagenes[:-1]).abs().mean().item()

    assert separacion(1000.0) < separacion(0.0)


def test_la_alineacion_no_tapa_una_unidad_muerta():
    """La alineacion tambien mueve la imagen. Si la comprobacion mirase el
    gradiente total, una unidad muerta pasaria desapercibida."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "canal", "capa": "layer4", "canal": 0}
    cfg["imagen"]["lote"] = 2
    cfg["regularizacion"]["alineacion"] = {"peso": 1.0, "capas": ["layer1"],
                                           "decaimiento": 2.0, "alcance": 4}
    modelo = _modelo_congelado()
    modelo.layer4.register_forward_hook(lambda _m, _e, s: s * 0)
    with pytest.raises(UnidadMuerta):
        maximizar_activacion(cfg, modelo=modelo)


def test_la_alineacion_suma_varias_capas(monkeypatch):
    """lucid alinea en dos capas a la vez. Si solo se usara la primera de la
    lista no habria error, solo una alineacion mas floja."""
    import src.interpretability.feature_viz as fv
    vistos = []
    original = fv.alineacion
    monkeypatch.setattr(fv, "alineacion",
                        lambda act, d, a: vistos.append(act.shape[1]) or original(act, d, a))
    cfg = copy.deepcopy(CFG)
    cfg["imagen"]["lote"] = 2
    cfg["optimizacion"]["pasos"] = 1
    cfg["regularizacion"]["alineacion"] = {"peso": 1.0, "capas": ["layer1", "layer2"],
                                           "decaimiento": 2.0, "alcance": 4}
    maximizar_activacion(cfg, modelo=_modelo_congelado())
    assert vistos == [64, 128]      # canales de layer1 y de layer2


def _con_canal_muerto(modelo, z):
    """layer4 con el canal z anulado pero aun en el grafo, como un ReLU."""
    mascara = torch.ones(1, 512, 1, 1)
    mascara[0, z] = 0
    modelo.layer4.register_forward_hook(lambda _m, _e, s: s * mascara)
    return modelo


def test_una_neurona_muerta_dentro_de_una_suma_se_detecta():
    """El canal 1 esta vivo y tapa al 0 si solo se mira la suma."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "suma", "capa": "layer4", "canales": [0, 1]}
    with pytest.raises(UnidadMuerta, match=r"canal\[0\]"):
        maximizar_activacion(cfg, modelo=_con_canal_muerto(_modelo_congelado(), 0))


def test_un_extremo_muerto_de_la_interpolacion_se_detecta():
    """Sin esto la primera imagen de la fila se quedaria en el ruido inicial
    y la interpolacion pareceria correcta."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "interpolacion", "capa": "layer4", "canales": [0, 1]}
    cfg["imagen"]["lote"] = 3
    with pytest.raises(UnidadMuerta, match="primera imagen"):
        maximizar_activacion(cfg, modelo=_con_canal_muerto(_modelo_congelado(), 0))


def test_los_hooks_se_quitan_al_terminar():
    """Si quedase uno enganchado, la siguiente llamada con el mismo modelo
    (la app lo reutiliza) capturaria de mas y gastaria memoria."""
    cfg = copy.deepcopy(CFG)
    cfg["imagen"]["lote"] = 2
    cfg["regularizacion"]["alineacion"] = {"peso": 1.0, "capas": ["layer1"],
                                           "decaimiento": 2.0, "alcance": 4}
    modelo = _modelo_congelado()
    maximizar_activacion(cfg, modelo=modelo)
    assert not modelo.fc._forward_hooks and not modelo.layer1._forward_hooks


def test_el_nombre_del_fichero_distingue_los_dos_objetivos():
    """Channel y Neuron sobre la misma capa y canal no pueden acabar en el
    mismo PNG, o uno pisa al otro sin avisar."""
    base = {"capa": "layer2", "canal": 42}
    canal = nombre_salida({**base, "tipo": "canal"})
    neurona = nombre_salida({**base, "tipo": "neurona", "x": None, "y": None})
    assert canal != neurona


def test_el_nombre_del_deepdream_no_depende_del_canal():
    """Layer no elige canal: lo que quede en el YAML no puede cambiar el
    nombre del PNG, o el mismo DeepDream se guardaria dos veces."""
    a = nombre_salida({"tipo": "capa", "capa": "layer3", "canal": 0})
    b = nombre_salida({"tipo": "capa", "capa": "layer3", "canal": 42})
    assert a == b == "layer3_capa"


def test_el_nombre_de_la_direccion_es_su_semilla():
    """Dos direcciones distintas no pueden pisarse el PNG, y el canal que
    quede en el YAML no puede cambiar el nombre de la misma direccion."""
    a = nombre_salida({"tipo": "direccion", "capa": "layer4", "direccion": 1, "canal": 0})
    b = nombre_salida({"tipo": "direccion", "capa": "layer4", "direccion": 1, "canal": 9})
    c = nombre_salida({"tipo": "direccion", "capa": "layer4", "direccion": 2, "canal": 0})
    assert a == b == "layer4_direccion_s1"
    assert a != c


def test_el_nombre_de_la_suma_lleva_todos_los_canales():
    """A+B y A solo no pueden pisarse el PNG."""
    assert nombre_salida({"tipo": "suma", "capa": "layer4", "canales": [12, 40]}) \
        == "layer4_suma_12+40"


def test_la_interpolacion_cambia_a_lo_largo_de_la_fila():
    """La primera imagen tiene que activar A mas que la ultima, y la ultima B
    mas que la primera: es lo que significa ir de una neurona a otra."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "interpolacion", "capa": "layer3", "canales": [1, 2]}
    cfg["imagen"]["lote"] = 3
    cfg["optimizacion"]["pasos"] = 24
    modelo = _modelo_congelado()
    fila, _ = maximizar_activacion(cfg, modelo=modelo)

    capturado = {}
    modelo.layer3.register_forward_hook(lambda _m, _e, s: capturado.__setitem__("a", s))
    ncfg = CFG["normalizacion"]
    media = torch.tensor(ncfg["media"]).view(1, 3, 1, 1)
    desv = torch.tensor(ncfg["desviacion"]).view(1, 3, 1, 1)
    with torch.no_grad():
        modelo((fila - media) / desv)
    canal_a = capturado["a"][:, 1].mean(dim=(1, 2))
    canal_b = capturado["a"][:, 2].mean(dim=(1, 2))
    assert canal_a[0] > canal_a[-1]
    assert canal_b[-1] > canal_b[0]


def test_el_nombre_de_la_interpolacion_lleva_los_dos_extremos():
    ocfg = {"tipo": "interpolacion", "capa": "layer4", "canales": [12, 40]}
    assert nombre_salida(ocfg) == "layer4_interpolacion_12-40"


def test_la_misma_seed_da_la_misma_imagen():
    """Sin esto, dos ejecuciones dan visualizaciones distintas y no se puede
    comparar una neurona consigo misma entre commits.

    El modelo se reutiliza a proposito: en uso real sale siempre del mismo
    checkpoint, asi que lo que se comprueba es que la aleatoriedad de la
    optimizacion (ruido inicial y transformaciones) esta bajo la seed.
    """
    modelo = _modelo_congelado()
    a, _ = maximizar_activacion(CFG, modelo=modelo)
    b, _ = maximizar_activacion(CFG, modelo=modelo)
    assert torch.allclose(a, b)


# --- Los controles de las figuras del articulo ---------------------------
#
# Hasta ahora ninguna de estas ramas se ejecutaba en los tests: la
# configuracion deja los regularizadores a 0, asi que `if peso_tv:` y sus
# hermanos nunca entraban. Un peso roto no se habria notado hasta que
# alguien moviera el deslizador en la aplicacion.

def _aspereza(img):
    """Cuanto cambia la imagen entre pixeles vecinos.

    Es lo que los penalizadores de frecuencia tienen que bajar, asi que
    sirve para comprobar que de verdad hacen algo.
    """
    return float((img[..., 1:, :] - img[..., :-1, :]).abs().mean()
                 + (img[..., :, 1:] - img[..., :, :-1]).abs().mean())


def _con(**cambios):
    cfg = copy.deepcopy(CFG)
    cfg["optimizacion"]["pasos"] = 24
    for ruta, valor in cambios.items():
        seccion, clave = ruta.split(".")
        cfg[seccion][clave] = valor
    return cfg


@pytest.mark.parametrize("peso", [{"regularizacion.l1": 5.0},
                                  {"regularizacion.tv": 5.0},
                                  {"regularizacion.blur": 50.0}])
def test_cada_penalizador_de_frecuencia_cambia_el_resultado(peso):
    """Un peso que no cambia nada es un deslizador roto en la interfaz."""
    modelo = _modelo_congelado()
    base, _ = maximizar_activacion(_con(), modelo=modelo)
    otra, _ = maximizar_activacion(_con(**peso), modelo=modelo)
    assert not torch.allclose(base, otra)


def test_la_variacion_total_suaviza():
    """De los tres penalizadores, el efecto de TV es el que se puede afirmar
    con seguridad: su definicion ES la aspereza que mide _aspereza."""
    modelo = _modelo_congelado()
    base, _ = maximizar_activacion(_con(), modelo=modelo)
    suave, _ = maximizar_activacion(_con(**{"regularizacion.tv": 20.0}),
                                    modelo=modelo)
    assert _aspereza(suave) < _aspereza(base)


def test_la_parametrizacion_por_pixeles_funciona():
    """El interruptor 'Decorrelated Space' en apagado. Sin esto no se puede
    reproducir la comparacion de la seccion de preconditioning."""
    imagen, historial = maximizar_activacion(
        _con(**{"imagen.parametrizacion": "pixel"}), modelo=_modelo_congelado())
    assert imagen.shape == (1, 3, TAM, TAM)
    assert imagen.min() >= 0.0 and imagen.max() <= 1.0
    assert sum(historial[-4:]) / 4 > sum(historial[:4]) / 4


def test_una_parametrizacion_desconocida_falla_y_dice_las_opciones():
    with pytest.raises(ValueError, match="fourier, pixel"):
        maximizar_activacion(_con(**{"imagen.parametrizacion": "wavelet"}),
                             modelo=_modelo_congelado())


def test_apagar_la_robustez_cambia_el_resultado():
    """El interruptor 'Transformation Robustness'. Apagarlo salta la funcion
    entera, que no es lo mismo que poner los valores a cero."""
    modelo = _modelo_congelado()
    base, _ = maximizar_activacion(_con(), modelo=modelo)
    cruda, _ = maximizar_activacion(_con(**{"robustez.activa": False}),
                                    modelo=modelo)
    assert not torch.allclose(base, cruda)


def test_la_deteccion_de_unidad_muerta_vale_para_las_dos_parametrizaciones():
    """_comprobar_gradiente miraba un tensor llamado 'espectro', que la
    parametrizacion por pixeles no tiene."""
    for cual in ("fourier", "pixel"):
        cfg = _con(**{"imagen.parametrizacion": cual})
        cfg["objetivo"] = {"tipo": "neurona", "capa": "layer4", "canal": 0}
        modelo = _modelo_congelado()
        modelo.layer4.register_forward_hook(lambda _m, _e, s: s * 0)
        with pytest.raises(UnidadMuerta):
            maximizar_activacion(cfg, modelo=modelo)


# --- Diversidad ------------------------------------------------------------

class _Run:
    """Sustituto de un run de W&B que guarda lo que se registra."""

    def __init__(self):
        self.metricas = []

    def log(self, m, step=None):
        self.metricas.append(m)


def _cfg_diversidad(peso, lote=3, pasos=4):
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "canal", "capa": "layer3", "canal": 1}
    cfg["imagen"]["lote"] = lote
    cfg["optimizacion"]["pasos"] = pasos
    cfg["regularizacion"]["diversidad"] = {"peso": peso, "capa": None}
    return cfg


def test_la_diversidad_de_un_lote_igual_es_maxima():
    """Tres imagenes identicas: cada par tiene coseno 1, y hay 3*2 pares
    ordenados entre 3 imagenes, asi que da 6 / 3 = 2 = lote - 1."""
    act = torch.rand(1, 4, 5, 5).repeat(3, 1, 1, 1)
    assert diversidad(act).item() == pytest.approx(2.0)


def test_la_diversidad_de_canales_disjuntos_es_cero():
    """Si cada imagen enciende canales distintos, sus Gram no comparten nada."""
    act = torch.zeros(2, 2, 3, 3)
    act[0, 0] = 1.0
    act[1, 1] = 1.0
    assert diversidad(act).item() == pytest.approx(0.0)


def test_la_diversidad_no_mira_donde_estan_las_cosas():
    """La Gram suma sobre posiciones: la misma imagen desplazada tiene el
    mismo "estilo". Es la idea que el articulo toma de la transferencia de
    estilo, y lo que la distingue de comparar pixeles."""
    a = torch.rand(1, 4, 6, 6)
    desplazada = torch.roll(a, shifts=(2, 3), dims=(2, 3))
    assert diversidad(torch.cat([a, desplazada])).item() == pytest.approx(1.0)


def test_la_diversidad_no_depende_de_la_escala():
    """Es una similitud coseno: activar mas fuerte no cuenta como distinto."""
    act = torch.rand(3, 4, 5, 5)
    assert torch.allclose(diversidad(act), diversidad(act * 7))


def test_la_diversidad_en_fc_falla_con_mensaje_util():
    with pytest.raises(ValueError, match="diversidad.capa"):
        diversidad(torch.randn(3, 10))


def test_la_diversidad_apagada_no_cambia_nada():
    """Lo que pidio el equipo: es algo que se PUEDE hacer, no un cambio en
    lo que ya habia. Con peso 0, lo mismo que sin la clave."""
    modelo = _modelo_congelado()
    cfg = _cfg_diversidad(0.0)
    sin_clave = copy.deepcopy(cfg)
    del sin_clave["regularizacion"]["diversidad"]
    a, _ = maximizar_activacion(cfg, modelo=modelo)
    b, _ = maximizar_activacion(sin_clave, modelo=modelo)
    assert torch.equal(a, b)


def test_la_diversidad_viene_apagada_en_el_yaml():
    from src.interpretability.feature_viz import RAIZ, cargar_config
    cfg = cargar_config(RAIZ / "configs" / "feature_viz.yaml")
    assert cfg["regularizacion"]["diversidad"]["peso"] == 0.0


def test_la_diversidad_con_una_imagen_no_hace_nada():
    modelo = _modelo_congelado()
    a, _ = maximizar_activacion(_cfg_diversidad(1000.0, lote=1), modelo=modelo)
    b, _ = maximizar_activacion(_cfg_diversidad(0.0, lote=1), modelo=modelo)
    assert torch.equal(a, b)


def test_la_diversidad_separa_las_imagenes():
    """El efecto que se busca: con el termino, las imagenes del lote acaban
    con Gram menos parecidas que sin el (partiendo del mismo ruido)."""
    def similitud_final(peso):
        modelo = _modelo_congelado()
        imagenes, _ = maximizar_activacion(_cfg_diversidad(peso, pasos=12), modelo=modelo)
        capturado = {}
        modelo.layer3.register_forward_hook(lambda _m, _e, s: capturado.__setitem__("a", s))
        ncfg = CFG["normalizacion"]
        media = torch.tensor(ncfg["media"]).view(1, 3, 1, 1)
        desv = torch.tensor(ncfg["desviacion"]).view(1, 3, 1, 1)
        with torch.no_grad():
            modelo((imagenes - media) / desv)
        return diversidad(capturado["a"]).item()

    assert similitud_final(100.0) < similitud_final(0.0)


def test_la_diversidad_apaga_alineacion_y_parte_compartida():
    """Las dos juntan las imagenes del lote; con diversidad se apagan, pero
    solo en esa ejecucion."""
    cfg = _cfg_diversidad(10.0)
    cfg["imagen"]["compartida"] = [[1, 4]]
    cfg["regularizacion"]["alineacion"] = {"peso": 100.0, "capas": ["layer1"],
                                           "decaimiento": 5.0, "alcance": 4}
    run = _Run()
    maximizar_activacion(cfg, run=run, modelo=_modelo_congelado())
    assert all("viz/diversidad" in m for m in run.metricas)
    assert not any("viz/alineacion" in m for m in run.metricas)
    # El cfg que llega no se modifica: la siguiente ejecucion sigue igual.
    assert cfg["imagen"]["compartida"] == [[1, 4]]


def test_la_interpolacion_sigue_alineando():
    """Comprobacion de que no se ha roto lo anterior: sin diversidad, la
    interpolacion conserva su alineacion."""
    cfg = copy.deepcopy(CFG)
    cfg["objetivo"] = {"tipo": "interpolacion", "capa": "layer3", "canales": [1, 2]}
    cfg["imagen"]["lote"] = 3
    cfg["regularizacion"]["alineacion"] = {"peso": 1.0, "capas": ["layer1"],
                                           "decaimiento": 5.0, "alcance": 4}
    run = _Run()
    maximizar_activacion(cfg, run=run, modelo=_modelo_congelado())
    assert all("viz/alineacion" in m for m in run.metricas)


def test_diversidad_e_interpolacion_no_se_combinan():
    cfg = _cfg_diversidad(10.0)
    cfg["objetivo"] = {"tipo": "interpolacion", "capa": "layer3", "canales": [1, 2]}
    with pytest.raises(ValueError, match="no se combinan"):
        maximizar_activacion(cfg, modelo=_modelo_congelado())


def test_la_diversidad_de_un_logit_se_mide_en_otra_capa():
    """Con un objetivo de clase hay que comparar en una conv: fc no tiene
    Gram. Con capa dada, funciona; sin ella, el error lo dice."""
    cfg = _cfg_diversidad(10.0)
    cfg["objetivo"] = {"tipo": "logit", "capa": "fc", "canal": 3}
    with pytest.raises(ValueError, match="diversidad.capa"):
        maximizar_activacion(cfg, modelo=_modelo_congelado())
    cfg["regularizacion"]["diversidad"]["capa"] = "layer4"
    imagenes, _ = maximizar_activacion(cfg, modelo=_modelo_congelado())
    assert imagenes.shape == (3, 3, TAM, TAM)


# --- Aplicacion: listar neuronas y ejemplos del dataset ----------------------

def _fotos(tmp_path, n=4):
    """Fotos sinteticas distintas: no hay dataset en CI."""
    from PIL import Image
    rutas = []
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        pixeles = (torch.rand(TAM, TAM, 3, generator=g) * 255).byte().numpy()
        ruta = tmp_path / f"foto_{i}.png"
        Image.fromarray(pixeles).save(ruta)
        rutas.append(ruta)
    return rutas


def _a_mano(ruta, capa, canal, modelo):
    """La activacion calculada sin pasar por la api, para comparar."""
    from src.interpretability.api import _preparar
    capturado = {}
    asa = resolver_modulo(modelo, capa).register_forward_hook(
        lambda _m, _e, s: capturado.__setitem__("a", s))
    with torch.no_grad():
        modelo(_preparar(ruta, CFG))
    asa.remove()
    return capturado["a"][0, canal].mean().item()


def test_listar_neuronas_sigue_dando_lo_mismo(tmp_path):
    """Se ha sacado el calculo a _por_canal: el ranking no puede cambiar."""
    from src.interpretability.api import listar_neuronas
    modelo = _modelo_congelado()
    ruta = _fotos(tmp_path, 1)[0]
    top = listar_neuronas(ruta, capas="layer3", top_k=5, modelo=modelo, cfg=CFG)
    for a in top:
        assert a.valor == pytest.approx(_a_mano(ruta, "layer3", a.canal, modelo), rel=1e-5)
    assert [a.valor for a in top] == sorted((a.valor for a in top), reverse=True)


def test_los_ejemplos_salen_ordenados_por_activacion(tmp_path):
    from src.interpretability.api import Activacion, ejemplos_del_dataset
    modelo = _modelo_congelado()
    rutas = _fotos(tmp_path)
    ejemplos = ejemplos_del_dataset(Activacion("layer3", 1, 0.0), tmp_path,
                                    top_k=3, modelo=modelo, cfg=CFG)
    esperado = sorted(rutas, key=lambda r: _a_mano(r, "layer3", 1, modelo), reverse=True)
    assert [e.foto for e in ejemplos] == esperado[:3]


def test_los_ejemplos_aceptan_una_lista_y_avisan_si_no_hay_fotos(tmp_path):
    from src.interpretability.api import Activacion, ejemplos_del_dataset
    modelo = _modelo_congelado()
    rutas = _fotos(tmp_path, 2)
    assert len(ejemplos_del_dataset(Activacion("fc", 3, 0.0), rutas, modelo=modelo, cfg=CFG)) == 2
    with pytest.raises(ValueError, match="get_data.py"):
        ejemplos_del_dataset(Activacion("fc", 3, 0.0), [], modelo=modelo, cfg=CFG)
