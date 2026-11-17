// Interfaz de visualizacion de caracteristicas.
//
// Es la v13 del diseno (Claude Design, "Inspector de neuronas") portada tal cual: el
// calculo de estilos de renderVals() es el mismo, linea por linea, en todo el diagrama.
// Lo que cambia es de donde salen los datos: en el diseno eran ficheros fijos; aqui los
// pide a la API de src/app/serve.py (analisis de la foto, atlas de mapas, mapa superpuesto)
// y la optimizacion es real, con su progreso, sus capturas y sus metricas paso a paso.
import { h, render, Component } from './vendor/preact.mjs';
import htm from './vendor/htm.mjs';
import { plantilla } from './plantilla.js';

const html = htm.bind(h);

// Lo que el diseno dibuja distinto segun el tamano del modelo: profundidad de cada pila
// y cuantas hojas lleva. El resto (capas, formas, bloques, atlas) llega del servidor.
const VISUAL = {
  pequeno: { d: [28, 40, 54, 72], hojas: [5, 7, 9, 12] },
  grande: { d: [40, 56, 76, 100], hojas: [8, 10, 13, 16] }
};
const BLOQUES_POR_DEFECTO = { resnet18: [2, 2, 2, 2], resnet50: [3, 4, 6, 3], resnet152: [3, 8, 36, 3] };
const SONDEO_MS = 250;

async function pedir(url, opciones) {
  const r = await fetch(url, opciones);
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(datos.error || r.statusText);
  return datos;
}

// Un tope "redondo" (1, 2, 2.5, 5 x 10^k) por encima de v: el eje solo cambia cuando la
// curva lo supera, en vez de reescalarse en cada paso mientras llegan los datos.
function topeRedondo(v) {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const f of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (f * e >= v) return f * e;
  return 10 * e;
}

class App extends Component {
  constructor(props) {
    super(props);
    this.cache = {};
    this.state = {
      info: null, foto: null, muestraSel: '', errorFoto: '', arrastrando: false,
      modelo: 'resnet18', cargando: true,
      mapa: true, mapaImg: null, mapaMax: 0, mapaClave: '',
      disp: 'cpu', trabajo: null, errorTexto: '',
      obj: 'canal',
      sel: 3, detalle: false, nivel2: false,
      datos: null, elegidos: [0, 0, 0, 0, 0],
      // Neuron: posicion (fila x, columna y) en cada etapa; por defecto el centro, como el codigo.
      pos: [{ x: 28, y: 28 }, { x: 14, y: 14 }, { x: 7, y: 7 }, { x: 3, y: 3 }],
      // Multiple channels: todos de la misma etapa. El ultimo marcado va delante.
      multi: { etapa: 3, canales: [] },
      hover: null,
      nombres: null,
      // Parametros: que secciones estan abiertas, los interruptores y los valores.
      abiertos: { opt: true },
      // Bloque elegido en cada etapa; sin entrada, el ultimo.
      bloque: {},
      // Paso bajo el raton en las graficas (compartido por las tres).
      cursor: null,
      flags: { robustez: true, fourier: true, wandb: false },
      params: null
    };
  }

  async componentDidMount() {
    try {
      const info = await pedir('/api/info');
      const v = info.valores;
      this.setState({
        info, nombres: info.especies, modelo: info.modelo_inicial, params: v,
        flags: { robustez: v.robustez, fourier: v.fourier, wandb: false }
      });
      this.ponerFoto(await pedir('/api/fotos/inicial', { method: 'POST' }));
    } catch (e) {
      this.setState({ cargando: false, errorFoto: 'No se pudo arrancar: ' + e.message });
    }
  }

  componentDidUpdate() {
    this.pedirMapa();
  }

  componentWillUnmount() {
    if (this.reloj) clearInterval(this.reloj);
  }

  // ---------------------------------------------------------------- fotos
  ponerFoto(foto, muestraSel = '') {
    this.setState({ foto, muestraSel, errorFoto: '', mapaImg: null, mapaClave: '' },
      () => this.cargarModelo(this.state.modelo, true));
  }

  async subir(fichero) {
    if (!fichero) return;
    try {
      const f = await pedir('/api/fotos', {
        method: 'POST', body: fichero,
        headers: { 'X-Nombre': encodeURIComponent(fichero.name), 'Content-Type': 'application/octet-stream' }
      });
      this.ponerFoto(f);
    } catch (e) {
      this.setState({ errorFoto: e.message });
    }
  }

  async elegirMuestra(nombre) {
    try {
      const f = nombre ? await pedir('/api/fotos/muestra', { method: 'POST', body: JSON.stringify({ nombre }) })
        : await pedir('/api/fotos/inicial', { method: 'POST' });
      this.ponerFoto(f, nombre);
    } catch (e) {
      this.setState({ errorFoto: e.message });
    }
  }

  // ---------------------------------------------------------------- modelo
  // Cambiar de modelo (o de foto) vuelve a la vista general. Con otro modelo se dejan los
  // canales por defecto del nuevo; con otra foto y el mismo modelo se conserva lo elegido.
  cargarModelo(arq, nuevaFoto) {
    const self = this;
    const s = this.state;
    if (!s.foto || (s.trabajo && s.trabajo.estado === 'corriendo' && arq !== s.modelo)) return;
    if (s.info && !s.info.modelos[arq].disponible) return;
    const mismo = arq === s.modelo && s.datos && s.datos.arquitectura === arq;
    function aplicar(d) {
      const st = self.state;
      if (!st.foto || st.foto.id !== fotoId || st.modelo !== arq) return;
      const cambios = { datos: d, cargando: false, detalle: false, nivel2: false };
      if (!mismo) {
        const el = d.etapas.map(function (e) { return e.elegido; });
        Object.assign(cambios, {
          elegidos: el, bloque: {}, multi: { etapa: 3, canales: [el[3]] },
          sel: st.obj === 'neurona' && st.sel === 4 ? 3 : st.sel,
          pos: d.etapas.slice(0, 4).map(function (e) { return { x: Math.floor(e.lado / 2), y: Math.floor(e.lado / 2) }; })
        });
      }
      self.setState(cambios);
    }
    const fotoId = s.foto.id, clave = fotoId + '|' + arq;
    if (this.cache[clave]) { this.setState({ modelo: arq }, function () { aplicar(self.cache[clave]); }); return; }
    // Con otra arquitectura, los datos viejos no casan con la estructura nueva: fuera mientras carga.
    this.setState({ modelo: arq, cargando: true, detalle: false, nivel2: false, datos: mismo ? s.datos : null });
    pedir('/api/fotos/' + fotoId + '/' + arq + '/analisis')
      .then(function (d) { self.cache[clave] = d; aplicar(d); })
      .catch(function (e) { self.setState({ cargando: false, errorFoto: e.message }); });
  }

  // Lo que el diseno tenia fijo por modelo, montado a partir del analisis que llega del servidor.
  modeloVisual(arq) {
    const st = this.state;
    const d = st.datos && st.datos.arquitectura === arq ? st.datos : null;
    const vis = arq === 'resnet18' ? VISUAL.pequeno : VISUAL.grande;
    const bloques = d ? d.etapas.slice(0, 4).map(function (e) { return e.bloques.length; }) : (BLOQUES_POR_DEFECTO[arq] || [2, 2, 2, 2]);
    return {
      etiqueta: st.info ? st.info.modelos[arq].etiqueta : arq,
      bloques: bloques,
      fc: d ? d.etapas[4] : { forma: '1081 clases', canales: 1081 },
      etapas: [0, 1, 2, 3].map(function (i) {
        const e = d ? d.etapas[i] : null;
        return {
          forma: e ? e.forma : '', d: vis.d[i], hojas: vis.hojas[i],
          cols: e ? e.atlas.cols : 8, filas: e ? e.atlas.filas : 8,
          hoja: 'hoja-l' + (i + 1), capa: e ? e.capa : 'layer' + (i + 1)
        };
      })
    };
  }

  cambiarObjetivo(o) {
    const cambios = { obj: o, nivel2: false };
    // DeepDream no entra en los canales; Neuron no tiene nada que hacer en fc.
    if (o === 'capa') cambios.detalle = false;
    if (o === 'neurona' && this.state.sel === 4) { cambios.sel = 3; cambios.detalle = false; }
    this.setState(cambios);
  }

  // ---------------------------------------------------------------- mapa superpuesto
  // Solo tiene sentido con el objetivo Channel sobre una capa convolucional: un canal, en
  // toda la foto. Con varios canales, una neurona suelta, la etapa entera o fc no hay un
  // mapa unico que ensenar, y el interruptor se bloquea (sin desaparecer).
  objetivoConMapa() {
    const st = this.state;
    return !!(st.datos && st.obj === 'canal' && st.sel < 4);
  }

  claveMapa() {
    const st = this.state;
    if (!this.objetivoConMapa() || !st.foto) return '';
    const e = st.datos.etapas[st.sel];
    const k = st.bloque[st.sel] == null ? e.bloques.length - 1 : st.bloque[st.sel];
    return [st.foto.id, st.modelo, e.bloques[k].capa, st.elegidos[st.sel]].join('|');
  }

  pedirMapa() {
    const clave = this.claveMapa();
    if (!clave || clave === this.mapaPedido) return;
    this.mapaPedido = clave;
    const [foto, arq, capa, canal] = clave.split('|');
    pedir('/api/fotos/' + foto + '/' + arq + '/mapa?capa=' + encodeURIComponent(capa) + '&canal=' + canal)
      .then((m) => { if (this.mapaPedido === clave) this.setState({ mapaImg: m.imagen, mapaMax: m.max, mapaClave: clave }); })
      .catch(() => {});
  }

  // ---------------------------------------------------------------- optimizacion
  peticionObjetivo() {
    const st = this.state;
    const e = st.datos.etapas[st.sel];
    const capaDe = (i) => {
      const et = st.datos.etapas[i];
      if (!et.bloques) return et.capa;
      return et.bloques[st.bloque[i] == null ? et.bloques.length - 1 : st.bloque[i]].capa;
    };
    if (st.obj === 'multi') {
      return { tipo: 'multi', capa: capaDe(st.multi.etapa), canales: st.multi.canales.slice(), etapa: st.multi.etapa };
    }
    const o = { tipo: st.obj, capa: capaDe(st.sel), canal: st.elegidos[st.sel], etapa: st.sel };
    if (st.obj === 'neurona') Object.assign(o, st.pos[st.sel]);
    if (e.capa === 'fc') o.capa = 'fc';
    return o;
  }

  async generar() {
    const st = this.state;
    if ((st.trabajo && st.trabajo.estado === 'corriendo') || !st.datos) return;
    const objetivo = this.peticionObjetivo();
    const p = st.params, fl = st.flags;
    const cuerpo = {
      foto: st.foto.id, modelo: st.modelo, dispositivo: st.disp, wandb: !!fl.wandb, objetivo,
      parametros: Object.assign({}, p, { robustez: fl.robustez, fourier: fl.fourier })
    };
    try {
      const { id } = await pedir('/api/trabajos', { method: 'POST', body: JSON.stringify(cuerpo) });
      const n = st.info.capturas;
      this.setState({
        errorTexto: '', cursor: null,
        trabajo: {
          id, estado: 'corriendo', paso: 0, pasos: p.pasos, segundos: 0, referencia: null,
          capturas: [], capturas_en: [], vista: null, vista_paso: 0, objetivo,
          metricas: { activacion: [], aspereza: [], cambio: [] }, nCapturas: n
        }
      });
      this.reloj = setInterval(() => this.sondear(), SONDEO_MS);
    } catch (e) {
      this.setState({ errorTexto: e.message });
    }
  }

  async sondear() {
    const t = this.state.trabajo;
    if (!t || this.sondeando) return;
    this.sondeando = true;
    try {
      const q = '?desde=' + t.metricas.activacion.length + '&capturas=' + t.capturas.length + '&vista=' + t.vista_paso;
      const r = await pedir('/api/trabajos/' + t.id + q);
      const actual = this.state.trabajo;
      if (!actual || actual.id !== t.id) return;
      const m = {};
      for (const k of ['activacion', 'aspereza', 'cambio']) m[k] = actual.metricas[k].slice(0, r.desde).concat(r.metricas[k]);
      const nuevo = Object.assign({}, actual, {
        estado: r.estado, paso: r.paso, pasos: r.pasos, segundos: r.segundos, referencia: r.referencia,
        capturas_en: r.capturas_en, capturas: actual.capturas.concat(r.capturas), metricas: m
      });
      if (r.vista) { nuevo.vista = r.vista; nuevo.vista_paso = r.vista_paso; }
      const cambios = { trabajo: nuevo };
      if (r.estado !== 'corriendo') {
        clearInterval(this.reloj);
        this.reloj = null;
        if (r.error) cambios.errorTexto = r.error;
      }
      this.setState(cambios);
    } catch (e) {
      clearInterval(this.reloj);
      this.reloj = null;
      this.setState({ errorTexto: 'Se perdio la conexion con el servidor: ' + e.message });
    } finally {
      this.sondeando = false;
    }
  }

  render() {
    const st = this.state;
    const v = this.renderVals();
    const foto = st.foto ? '/api/fotos/' + st.foto.id : '';
    const vars = st.foto ? {
      '--foto': 'url(' + foto + '/foto.png)', '--rgb': 'url(' + foto + '/rgb.png)',
      '--atlas-l1': 'url(' + foto + '/' + st.modelo + '/atlas/0.png)',
      '--atlas-l2': 'url(' + foto + '/' + st.modelo + '/atlas/1.png)',
      '--atlas-l3': 'url(' + foto + '/' + st.modelo + '/atlas/2.png)',
      '--atlas-l4': 'url(' + foto + '/' + st.modelo + '/atlas/3.png)'
    } : {};
    // Las variables solo se fijan cuando ya hay analisis del modelo: si no, la pagina pediria
    // los atlas antes de que existan.
    if (!st.datos || st.datos.arquitectura !== st.modelo) { delete vars['--atlas-l1']; delete vars['--atlas-l2']; delete vars['--atlas-l3']; delete vars['--atlas-l4']; }
    return html`<div style=${vars}>${plantilla(v)}</div>`;
  }

  renderVals() {
    var self = this;
    var st = this.state;
    var mapa = st.mapa;
    var obj = st.obj;

    // ===================================================== objetivo
    var OPCIONES = [
      { id: 'canal', etiqueta: 'Channel' },
      { id: 'neurona', etiqueta: 'Neuron' },
      { id: 'capa', etiqueta: 'Layer (DeepDream)' },
      { id: 'multi', etiqueta: 'Multiple channels' }
    ];
    var objetivos = OPCIONES.map(function (o, k) {
      var activo = o.id === obj;
      var radio = k === 0 ? 'border-radius:6px 0 0 6px;' : k === OPCIONES.length - 1 ? 'border-radius:0 6px 6px 0;' : 'border-radius:0;';
      return {
        etiqueta: o.etiqueta,
        aria: activo ? 'true' : 'false',
        estilo: 'font:inherit;font-size:14px;padding:8px 14px;min-height:40px;cursor:pointer;' + radio +
          (k > 0 ? 'margin-left:-1px;' : '') +
          'border:1px solid ' + (activo ? '#0f6f5c' : '#dddcd3') + ';background:' + (activo ? '#e3efeb' : '#fbfbf9') +
          ';color:' + (activo ? '#0f6f5c' : '#1b1c19') + (activo ? ';position:relative;z-index:1' : ''),
        pulsar: function () { self.cambiarObjetivo(o.id); }
      };
    });

    // ===================================================== la red en 3D
    var CY = 175;
    var GIRO = 'rotateX(-18deg) rotateY(-34deg)';
    var ENTRADA = { w: 120, h: 120, d: 18, cx: 80 };
    var MOD = this.modeloVisual(st.modelo);
    function bloqueDe(i, n) { return st.bloque[i] == null ? n - 1 : st.bloque[i]; }
    function datosEtapa(i) {
      var e = st.datos.etapas[i];
      return e.bloques ? Object.assign({}, e, e.bloques[bloqueDe(i, e.bloques.length)]) : e;
    }
    // Lo que no cambia entre modelos: la resolucion de cada etapa y su sitio en el diagrama.
    var BASE = [
      { nombre: 'layer1', w: 104, h: 104, cx: 245, lado: 56 },
      { nombre: 'layer2', w: 84, h: 84, cx: 395, lado: 28 },
      { nombre: 'layer3', w: 66, h: 66, cx: 530, lado: 14 },
      { nombre: 'layer4', w: 50, h: 50, cx: 655, lado: 7 }
    ];
    var ETAPAS = BASE.map(function (b, i) {
      var nb = MOD.bloques[i], k = bloqueDe(i, nb);
      var e = Object.assign({}, b, MOD.etapas[i], { nBloques: nb, bloqueSel: k,
        bloques: k === nb - 1 ? nb + ' bloques' : 'bloque ' + (k + 1) + ' de ' + nb });
      if (st.datos && st.datos.etapas[i].bloques) e.capa = st.datos.etapas[i].bloques[k].capa;
      return e;
    }).concat([{ capa: 'fc', nombre: 'fc', forma: MOD.fc.forma, w: 14, h: 120, d: 14, cx: 780, lado: 0, bloques: '' }]);

    // Solo visual: detras de la pila de canales (la salida del ultimo bloque) va una lamina fina por
    // cada bloque anterior, asi el grosor de cada etapa crece con la profundidad del modelo.
    var PASO_PLACA = 4, EXTRA_MAX = 50;
    ETAPAS.forEach(function (b) {
      b.dc = b.d;                                   // profundidad de la pila de canales
      b.placas = b.hojas ? b.nBloques - 1 : 0;
      b.extra = Math.min(EXTRA_MAX, b.placas * PASO_PLACA);
      b.d = b.dc + b.extra;                         // profundidad total, la que ocupa en el diagrama
    });
    // Con etapas mas gruesas no caben en sitios fijos: se reparten entre la entrada y fc con huecos iguales.
    function media(b) { return (b.w * 0.83 + b.d * 0.56) / 2; }
    var desdeX = ENTRADA.cx + media(ENTRADA), hastaX = ETAPAS[4].cx - media(ETAPAS[4]);
    var ocupado = 0;
    for (var q = 0; q < 4; q++) ocupado += 2 * media(ETAPAS[q]);
    var huecoX = (hastaX - desdeX - ocupado) / 5, xAcum = desdeX;
    for (q = 0; q < 4; q++) {
      xAcum += huecoX + media(ETAPAS[q]);
      ETAPAS[q].cx = Math.round(xAcum);
      xAcum += media(ETAPAS[q]);
    }
    var detalle = st.detalle && obj !== 'capa';
    var nivel2 = detalle && obj === 'neurona' && st.nivel2;
    var sel = st.sel;

    // El canal que va delante en la pila de cada etapa.
    function frontal(i) {
      if (obj === 'multi' && st.multi.etapa === i && st.multi.canales.length) {
        return st.multi.canales[st.multi.canales.length - 1];
      }
      return st.elegidos[i];
    }
    function borde(marcada) { return marcada ? '1px solid #3ebd9d' : '1px solid rgba(237,239,233,.32)'; }
    // Con DeepDream la elegida se resalta entera; con el resto, la hoja de delante.
    function pila(b, canales, marcada, claseDe, fondoDe, z0) {
      var n = canales.length;
      var paso = n > 1 ? b.d / (n - 1) : 0;
      var entera = marcada && obj === 'capa';
      return canales.map(function (canal, k) {
        var z = (z0 == null ? b.d / 2 : z0) - k * paso;
        var luz = Math.max(0.42, 1 - k * Math.min(0.07, 0.58 / Math.max(1, n - 1)));
        var resaltada = marcada && (k === 0 || entera);
        return {
          clase: 'hoja ' + claseDe(k),
          estilo: 'left:0;top:0;width:' + b.w + 'px;height:' + b.h + 'px;transform:translateZ(' + z.toFixed(1) + 'px);' +
            fondoDe(canal, k) + 'border:' + borde(resaltada) + ';filter:brightness(' + luz.toFixed(2) + ')' +
            (marcada && k === 0 ? ';box-shadow:0 0 18px rgba(62,189,157,.45)' : '')
        };
      });
    }
    function recorte(b, canal, lado) {
      var c = canal % b.cols, f = Math.floor(canal / b.cols) + (b.bloqueSel || 0) * b.filas;
      return 'background-size:' + (b.cols * lado) + 'px ' + (b.filas * (b.nBloques || 1) * lado) + 'px;background-position:' + (-c * lado) + 'px ' + (-f * lado) + 'px;';
    }
    function cubosFc(b, marcada, i) {
      var N = 12, alto = 8, paso = 10;
      var e = st.datos ? datosEtapa(i) : null;
      var elegido = frontal(i);
      var lista = [elegido];
      for (var j = 1; lista.length < N; j++) {
        var c = (elegido + j * 97) % MOD.fc.canales;
        if (lista.indexOf(c) < 0) lista.push(c);
      }
      var w = b.w, d = b.d, caras = [];
      lista.forEach(function (clase, k) {
        var color = e ? e.colores[clase] : '#3a3f39';
        var t = k * paso;
        var resaltada = marcada && (k === 0 || obj === 'capa');
        var br = resaltada ? '1px solid #3ebd9d' : '1px solid rgba(237,239,233,.2)';
        var brillo = (marcada && k === 0) ? ';box-shadow:0 0 14px rgba(62,189,157,.5)' : '';
        caras.push(
          { clase: 'cara', estilo: 'left:0;top:' + t + 'px;width:' + w + 'px;height:' + alto + 'px;transform:rotateY(180deg) translateZ(' + d / 2 + 'px);background:' + color + ';filter:brightness(.6);border:' + br },
          { clase: 'cara', estilo: 'left:' + (w - d) / 2 + 'px;top:' + t + 'px;width:' + d + 'px;height:' + alto + 'px;transform:rotateY(-90deg) translateZ(' + w / 2 + 'px);background:' + color + ';filter:brightness(.72);border:' + br },
          { clase: 'cara', estilo: 'left:' + (w - d) / 2 + 'px;top:' + t + 'px;width:' + d + 'px;height:' + alto + 'px;transform:rotateY(90deg) translateZ(' + w / 2 + 'px);background:' + color + ';filter:brightness(.72);border:' + br },
          { clase: 'cara', estilo: 'left:0;top:' + (t + (alto - d) / 2) + 'px;width:' + w + 'px;height:' + d + 'px;transform:rotateX(-90deg) translateZ(' + alto / 2 + 'px);background:' + color + ';filter:brightness(.55);border:' + br },
          { clase: 'cara', estilo: 'left:0;top:' + (t + (alto - d) / 2) + 'px;width:' + w + 'px;height:' + d + 'px;transform:rotateX(90deg) translateZ(' + alto / 2 + 'px);background:' + color + ';filter:brightness(1.18);border:' + br },
          { clase: 'cara', estilo: 'left:0;top:' + t + 'px;width:' + w + 'px;height:' + alto + 'px;transform:translateZ(' + d / 2 + 'px);background:' + color + ';border:' + br + brillo }
        );
      });
      return caras;
    }
    function caja(b) {
      var bw = b.w + b.d + 24, bh = b.h + b.d + 24;
      return {
        pos: 'position:absolute;left:' + (b.cx - bw / 2) + 'px;top:' + (CY - bh / 2) + 'px;width:' + bw + 'px;height:' + bh + 'px;perspective:900px;',
        cubo: 'position:absolute;display:block;left:' + (bw - b.w) / 2 + 'px;top:' + (bh - b.h) / 2 + 'px;width:' + b.w + 'px;height:' + b.h + 'px;transform-style:preserve-3d;transform:' + GIRO
      };
    }
    function rotulo(b, visible) {
      return 'position:absolute;left:' + b.cx + 'px;top:268px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;gap:2px;font-size:13px;white-space:nowrap;transition:opacity .4s ease;opacity:' + (visible ? 1 : 0);
    }

    var cE = caja(ENTRADA);
    var entrada = {
      posEstilo: cE.pos + 'transition:opacity .5s ease;opacity:' + (detalle ? 0.06 : 1),
      cuboEstilo: cE.cubo,
      caras: pila(ENTRADA, [-1, 0, 1, 2], false,
        function (k) { return k === 0 ? 'hoja-foto' : 'hoja-rgb'; },
        function (canal) {
          return canal < 0 ? '' : 'background-size:' + (3 * ENTRADA.w) + 'px ' + ENTRADA.h + 'px;background-position:' + (-canal * ENTRADA.w) + 'px 0;';
        }),
      rotuloEstilo: rotulo(ENTRADA, !detalle)
    };

    var bloques = ETAPAS.map(function (b, i) {
      var c = caja(b);
      var elegida = i === sel;
      var inhabilitada = obj === 'neurona' && b.lado === 0;
      var apagada = (detalle && !elegida) || inhabilitada;
      var caras;
      if (b.hojas) {
        var delante = frontal(i);
        var e = st.datos ? datosEtapa(i) : null;
        // Un modelo mas profundo lleva mas hojas (una mas por bloque extra, hasta 12), todas con su mapa:
        // primero las de fondo y, si no llegan, los canales mas activos de la etapa.
        var total = b.hojas + Math.min(b.placas, 12);
        var fondo = [];
        if (e) {
          var usados = {}; usados[delante] = true;
          var porActividad = e.valores.map(function (v, k) { return k; })
            .sort(function (p, q) { return e.valores[q] - e.valores[p]; }).slice(0, total * 2);
          var candidatos = (e.fondo || []).concat(porActividad);
          for (var q = 0; q < candidatos.length && fondo.length < total - 1; q++) {
            if (!usados[candidatos[q]]) { usados[candidatos[q]] = true; fondo.push(candidatos[q]); }
          }
        }
        caras = pila(b, [delante].concat(fondo), elegida,
          function () { return b.hoja; },
          function (canal) { return recorte(b, canal, b.w); });
      } else {
        caras = cubosFc(b, elegida, i);
      }
      return {
        nombre: b.nombre, forma: inhabilitada ? 'sin mapa espacial' : b.forma, bloques: b.bloques,
        aria: b.nombre + ', ' + b.forma + (obj === 'capa' ? '. Optimizar la etapa entera' : '. Ver sus canales'),
        elegidaAria: elegida ? 'true' : 'false',
        inhabilitadaAria: inhabilitada ? 'true' : 'false',
        posEstilo: c.pos + 'border:none;background:none;padding:0;margin:0;cursor:' + (inhabilitada ? 'not-allowed' : 'pointer') +
          ';opacity:' + (detalle && !elegida ? 0.06 : inhabilitada ? 0.3 : 1) + ';pointer-events:' + (detalle || inhabilitada ? 'none' : 'auto'),
        cuboEstilo: c.cubo,
        caras: caras,
        rotuloEstilo: rotulo(b, !detalle),
        nombreEstilo: elegida && !inhabilitada ? 'font-weight:600;color:#3ebd9d' : 'font-weight:600;color:#edefe9',
        pulsar: function () {
          if (inhabilitada) return;
          if (self.state.obj === 'capa') self.setState({ sel: i, detalle: false });
          else self.setState({ sel: i, detalle: true, nivel2: false });
        }
      };
    });
    var bordes = [ENTRADA].concat(ETAPAS).map(function (b) {
      var media = (b.w * 0.83 + b.d * 0.56) / 2;
      return { izq: b.cx - media, der: b.cx + media };
    });
    var flechas = [];
    for (var k = 0; k < bordes.length - 1; k++) {
      flechas.push({ x1: (bordes[k].der + 6).toFixed(0), x2: (bordes[k + 1].izq - 8).toFixed(0) });
    }
    var salidaX = (bordes[bordes.length - 1].der + 10).toFixed(0);

    // Camara: nivel 1 acerca el bloque; nivel 2 (Neuron) se acerca todavia mas.
    var bS = ETAPAS[sel];
    // Con pilas profundas (ResNet-50/152) el zoom se acota para que la pila quepa entera a la izquierda del panel.
    var anchoPila = bS.w * 0.83 + bS.d * 0.56 + 24;
    var zoom = Math.max(1.3, Math.min(3.2, 170 / Math.max(bS.w, bS.h), 300 / anchoPila)) * (nivel2 ? 1.5 : 1);
    var tx = 165 - zoom * bS.cx, ty = 185 - zoom * CY;
    var camaraEstilo = 'position:absolute;inset:0;transform-origin:0 0;transition:transform .9s cubic-bezier(.2,.75,.2,1),opacity .6s ease;transform:' +
      (detalle ? 'translate(' + tx.toFixed(1) + 'px,' + ty.toFixed(1) + 'px) scale(' + zoom.toFixed(2) + ')' : 'none');
    // Se oscurece con un velo por encima: bajar la opacity de la camara obliga a recomponer toda la escena 3D de golpe.
    var veloEstilo = 'position:absolute;top:-8px;bottom:-8px;left:-2000px;right:-2000px;background:#141513;pointer-events:none;transition:opacity .4s ease;opacity:' + (nivel2 || st.cargando ? 0.62 : 0);

    // ---- nivel 1: la rejilla de canales
    var etapa = st.datos ? datosEtapa(sel) : null;
    var celdas = [], rejillaEstilo = '', panelMeta = '', panelInfo = '', leyendaMin = '', leyendaMax = '';
    var marcados = obj === 'multi' && st.multi.etapa === sel ? st.multi.canales : [];
    function especie(c) {
      var n = st.nombres && st.nombres[c];
      return n ? n[0] + (n[1] ? ' ' + n[1] : '') : '';
    }
    function fmt(v) { return (v >= 0 ? '+' : '') + v.toFixed(3); }
    var esFc = bS.lado === 0;
    if (etapa) {
      var n = etapa.canales, cols = etapa.cols, filas = Math.ceil(n / cols);
      var hueco = filas > 20 ? 1 : 2;   // 2048 canales: 32 filas, el hueco de 2 px no cabe
      var anchoCelda = (520 - hueco * (cols - 1)) / cols;
      var alto = Math.max(5, Math.min(anchoCelda, 186 / filas - hueco));   // 186: deja sitio a la tira de bloques
      var unico = st.elegidos[sel];
      var ultimo = marcados.length ? marcados[marcados.length - 1] : -1;
      // Al bajar al nivel 2 las celdas no se animan una a una: el panel entero se aleja, y vuelve igual.
      var visible = detalle;
      celdas = etapa.colores.map(function (col, i) {
        var fila = Math.floor(i / cols);
        var retraso = visible ? (0.45 + fila * 0.03).toFixed(2) : '0';
        var marcada = obj === 'multi' ? marcados.indexOf(i) >= 0 : i === unico;
        var anillo = !marcada ? '' :
          (obj === 'multi' && i !== ultimo ? 'box-shadow:0 0 0 2px rgba(62,189,157,.75);' : 'box-shadow:0 0 0 2px #3ebd9d,0 0 8px rgba(62,189,157,.8);') + 'position:relative;z-index:2;';
        return {
          titulo: (esFc ? 'clase ' + i + (especie(i) ? ' · ' + especie(i) : '') : 'canal ' + i) + ' · ' + fmt(etapa.valores[i]),
          aria: marcada ? 'true' : 'false',
          estilo: 'display:block;border:none;padding:0;margin:0;border-radius:1.5px;cursor:pointer;background:' + col + ';' + anillo +
            'transition:opacity .35s ease ' + retraso + 's,transform .35s ease ' + retraso + 's;' +
            (visible ? 'opacity:1;transform:none' : 'opacity:0;transform:scale(.3)'),
          pulsar: function () {
            var s = self.state;
            if (s.obj === 'multi') {
              // Marcar o desmarcar. Si se marca en otra etapa, se empieza de cero.
              var lista = s.multi.etapa === s.sel ? s.multi.canales.slice() : [];
              var p = lista.indexOf(i);
              if (p >= 0) lista.splice(p, 1); else lista.push(i);
              self.setState({ multi: { etapa: s.sel, canales: lista } });
            } else {
              var e = s.elegidos.slice();
              e[s.sel] = i;
              self.setState(s.obj === 'neurona' ? { elegidos: e, nivel2: true } : { elegidos: e });
            }
          }
        };
      });
      rejillaEstilo = 'display:grid;grid-template-columns:repeat(' + cols + ',minmax(0,1fr));grid-auto-rows:' + alto.toFixed(1) + 'px;gap:' + hueco + 'px;width:520px';
      panelMeta = esFc ? '1081 clases, sin extension espacial' : n + ' canales, mapa de ' + etapa.lado + '×' + etapa.lado;
      if (obj === 'multi') {
        panelInfo = marcados.length === 0 ? 'Pulsa los canales que quieras sumar' :
          marcados.length + (marcados.length === 1 ? (esFc ? ' clase: ' : ' canal: ') : (esFc ? ' clases: ' : ' canales: ')) + marcados.join(', ');
      } else if (obj === 'neurona') {
        panelInfo = 'Pulsa un canal para elegir una de sus posiciones';
      } else {
        panelInfo = (esFc ? 'Clase ' : 'Canal ') + unico + ' · ' + (esFc ? 'logit ' : 'activacion media ') + fmt(etapa.valores[unico]);
      }
      leyendaMin = fmt(etapa.lo);
      leyendaMax = fmt(etapa.hi);
    }
    // fc: la especie de la clase elegida (o, con varias, de la ultima marcada) va en la cabecera del panel.
    var claseFc = !esFc ? -1 : obj === 'multi' ? (marcados.length ? marcados[marcados.length - 1] : -1) : st.elegidos[sel];
    var nombreFc = claseFc >= 0 && st.nombres ? st.nombres[claseFc] : null;
    // ---- la tira de bloques de la etapa elegida
    var tira = [], tiraTexto = '', hayTira = !!(bS.hojas && bS.nBloques > 1 && st.datos);
    if (hayTira) {
      var lb = st.datos.etapas[sel].bloques;
      tira = lb.map(function (blq, k) {
        var activo = k === bS.bloqueSel;
        return {
          titulo: 'Bloque ' + (k + 1) + ' de ' + lb.length + ' · ' + blq.capa,
          pulsado: activo ? 'true' : 'false',
          estilo: 'flex:1 1 0;min-width:0;height:16px;padding:0;margin:0;border:none;border-radius:2px;cursor:pointer;transition:background .15s ease;background:' +
            (activo ? '#3ebd9d' : '#3a3f39'),
          pulsar: function () {
            var s = self.state;
            var bq = Object.assign({}, s.bloque); bq[s.sel] = k;
            var el = s.elegidos.slice(); el[s.sel] = s.datos.etapas[s.sel].bloques[k].elegido;
            var cambios = { bloque: bq, elegidos: el };
            if (s.multi.etapa === s.sel) cambios.multi = { etapa: s.sel, canales: [el[s.sel]] };
            self.setState(cambios);
          }
        };
      });
      tiraTexto = (bS.bloqueSel + 1) + ' / ' + lb.length;
    }
    var tiraEstilo = 'flex:1;min-width:0;display:flex;gap:' + (tira.length > 20 ? 1 : 2) + 'px';
    var abierto1 = detalle && !nivel2;
    var panelEstilo = 'position:absolute;left:330px;top:34px;width:520px;display:flex;flex-direction:column;gap:12px;transform-origin:0 50%;' +
      'transition:opacity .45s ease ' + (abierto1 ? '.35s' : '0s') + ',transform .7s cubic-bezier(.2,.75,.2,1) ' + (abierto1 ? '.3s' : '0s') + ';' +
      (abierto1 ? 'opacity:1;transform:none;pointer-events:auto' :
        nivel2 ? 'opacity:0;transform:scale(1.18);pointer-events:none' :
        'opacity:0;transform:perspective(900px) rotateY(32deg) translateX(-150px) scale(.5);pointer-events:none');

    // ---- nivel 2 (Neuron): el mapa del canal elegido, posicion a posicion
    var LADO_MAPA = 252;
    // Un solo boton encima del mapa en vez de L×L botones (hasta 3136 en layer1): el mapa ya esta
    // montado antes de pulsar el canal y la transicion solo mueve opacity y transform.
    var mapaEstilo = 'display:none', mapaClase = '', mapaPos = '', mapaMeta = '', mapaCanal = '';
    var marcaEstilo = 'display:none', hoverEstilo = 'display:none', selectorAria = '';
    var campoEstilo = 'display:none', campoTexto = '', campoAria = '';
    if (!esFc && bS.hojas && st.datos) {
      var canalN = st.elegidos[sel];
      var pos = st.pos[sel];
      var L = bS.lado;
      var celda = LADO_MAPA / L;
      var cuadro = function (p) {
        return 'position:absolute;pointer-events:none;left:' + (p.y * celda).toFixed(2) + 'px;top:' + (p.x * celda).toFixed(2) +
          'px;width:' + celda.toFixed(2) + 'px;height:' + celda.toFixed(2) + 'px;';
      };
      mapaCanal = String(canalN);
      mapaClase = bS.hoja;
      mapaMeta = 'mapa de ' + L + '×' + L + ' · pulsa una posicion';
      mapaPos = 'Fila ' + pos.x + ', columna ' + pos.y;
      // El recorte se pinta a su tamano real (L×L) y se amplia con transform, sin reescalar el atlas entero.
      mapaEstilo = 'left:0;top:0;width:' + L + 'px;height:' + L + 'px;' + recorte(bS, canalN, L) +
        // Vecino mas cercano: cada posicion del mapa es un cuadrado nitido, sin suavizado entre celdas.
        'transform:scale(' + celda.toFixed(4) + ');transform-origin:0 0;image-rendering:pixelated;background-repeat:no-repeat';
      marcaEstilo = cuadro(pos) + 'box-shadow:0 0 0 2px #3ebd9d,0 0 10px rgba(62,189,157,.9);z-index:2';
      var hv = st.hover;
      hoverEstilo = hv && nivel2 && hv.x < L && hv.y < L ? cuadro(hv) + 'box-shadow:0 0 0 1px rgba(237,239,233,.85);z-index:1' : 'display:none';
      selectorAria = 'Mapa de ' + L + '×' + L + ' del canal ' + canalN + '. Elegida la fila ' + pos.x + ', columna ' + pos.y + '. Usa las flechas para moverla.';
      // Campo receptivo: sigue al raton sobre el mapa y, si no, a la posicion elegida.
      var CR = st.datos.campos;
      var rfLado = CR.rf[sel][bS.bloqueSel], salto = CR.j[sel], ini = CR.inicio[sel][bS.bloqueSel];
      var pc = hv && nivel2 && hv.x < L && hv.y < L ? hv : pos;
      var cxF = ini + pc.y * salto, cyF = ini + pc.x * salto;      // fila = vertical, columna = horizontal
      var x0 = Math.max(0, cxF - rfLado / 2), x1 = Math.min(224, cxF + rfLado / 2);
      var y0 = Math.max(0, cyF - rfLado / 2), y1 = Math.min(224, cyF + rfLado / 2);
      var esc = 160 / 224, entero = x0 <= 0 && y0 <= 0 && x1 >= 224 && y1 >= 224;
      campoEstilo = 'position:absolute;box-sizing:border-box;border:2px solid #3ebd9d;border-radius:2px;' +
        'box-shadow:0 0 0 400px rgba(20,21,19,.62);transition:left .2s ease,top .2s ease,width .2s ease,height .2s ease;' +
        'left:' + (x0 * esc).toFixed(1) + 'px;top:' + (y0 * esc).toFixed(1) + 'px;width:' + ((x1 - x0) * esc).toFixed(1) + 'px;height:' + ((y1 - y0) * esc).toFixed(1) + 'px';
      campoTexto = 'Campo receptivo: ' + rfLado + '×' + rfLado + ' px de una foto de 224' +
        (entero ? '. Ya cubre la foto entera.' : rfLado > 224 ? '. Es mas grande que la foto: solo se ve la parte que cae dentro.' : '.');
      campoAria = 'Campo receptivo de la fila ' + pc.x + ', columna ' + pc.y + ': ' + rfLado + ' por ' + rfLado + ' pixeles de la foto';
    }
    function limitar(v, n) { return Math.max(0, Math.min(n - 1, v)); }
    function celdaDe(ev) {
      var r = ev.currentTarget.getBoundingClientRect();
      var n = ETAPAS[self.state.sel].lado;
      return {
        x: limitar(Math.floor((ev.clientY - r.top) / r.height * n), n),
        y: limitar(Math.floor((ev.clientX - r.left) / r.width * n), n)
      };
    }
    function fijarPos(p) {
      var lista = self.state.pos.slice();
      lista[self.state.sel] = p;
      self.setState({ pos: lista });
    }
    var pulsarMapa = function (ev) {
      if (ev.detail === 0) return;   // Enter/espacio: la posicion ya se mueve con las flechas
      fijarPos(celdaDe(ev));
    };
    var moverMapa = function (ev) {
      var c = celdaDe(ev), h = self.state.hover;
      if (!h || h.x !== c.x || h.y !== c.y) self.setState({ hover: c });
    };
    var salirMapa = function () { if (self.state.hover) self.setState({ hover: null }); };
    var teclaMapa = function (ev) {
      var d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[ev.key];
      if (!d) return;
      ev.preventDefault();
      var n = ETAPAS[self.state.sel].lado, p = self.state.pos[self.state.sel];
      fijarPos({ x: limitar(p.x + d[0], n), y: limitar(p.y + d[1], n) });
    };
    // Mismos tiempos y curva que la entrada al nivel 1.
    var mapaPanelEstilo = 'position:absolute;left:330px;top:34px;width:520px;display:flex;flex-direction:column;gap:12px;transform-origin:0 50%;' +
      'transition:opacity .45s ease ' + (nivel2 ? '.35s' : '0s') + ',transform .7s cubic-bezier(.2,.75,.2,1) ' + (nivel2 ? '.3s' : '0s') + ';' +
      (nivel2 ? 'opacity:1;transform:none;pointer-events:auto' : 'opacity:0;transform:scale(.82);pointer-events:none');

    // ---- lo que se va a optimizar, siempre a la vista en la vista general
    var resumen;
    if (obj === 'capa') {
      resumen = bS.capa + ' · la etapa entera';
    } else if (obj === 'neurona') {
      var pR = st.pos[sel] || { x: 0, y: 0 };
      resumen = bS.capa + ' · canal ' + st.elegidos[sel] + ' · fila ' + pR.x + ', columna ' + pR.y;
    } else if (obj === 'multi') {
      var lista = st.multi.canales;
      resumen = lista.length === 0 ? 'Ningun canal marcado' :
        ETAPAS[st.multi.etapa].capa + ' · ' + (lista.length === 1 ? 'canal ' : 'canales ') + lista.join(', ');
    } else {
      resumen = bS.capa + ' · ' + (esFc ? 'clase ' : 'canal ') + st.elegidos[sel] +
        (esFc && especie(st.elegidos[sel]) ? ' · ' + st.nombres[st.elegidos[sel]][0] : '');
    }
    var PISTAS = {
      canal: 'Pulsa una etapa para ver sus canales',
      neurona: 'Pulsa una etapa, despues un canal y despues una posicion de su mapa',
      capa: 'Pulsa una etapa para optimizarla entera',
      multi: 'Pulsa una etapa y marca los canales que quieras sumar'
    };

    // ===================================================== parametros, en secciones plegables
    var fl = st.flags;
    var P = st.params || {}, RANGOS = st.info ? st.info.rangos : {};
    function tipo(t) { return { esSlider: t === 's', esToggle: t === 't', esTexto: t === 'x', esEnlace: t === 'e' }; }
    function envoltorio(apagado) {
      return 'display:flex;flex-direction:column;gap:4px;transition:opacity .2s ease;' + (apagado ? 'opacity:.4;pointer-events:none' : '');
    }
    // Un deslizador del diseno, ahora con valor de verdad: la posicion sale del rango de
    // configs/app.yaml (lineal o logaritmica) y el input range invisible de encima lo mueve.
    function slider(clave, etiqueta, formato, ayuda, apagado) {
      var r = RANGOS[clave] || { min: 0, max: 1 };
      var log = r.escala === 'log';
      var a = log ? Math.log(r.min) : r.min, b = log ? Math.log(r.max) : r.max;
      var valor = P[clave] != null ? P[clave] : r.min;
      var posicion = Math.max(0, Math.min(1, ((log ? Math.log(valor) : valor) - a) / (b - a)));
      return Object.assign(tipo('s'), {
        etiqueta: etiqueta, valor: formato(valor), min: formato(r.min), max: formato(r.max), ayuda: ayuda, envoltorio: envoltorio(apagado),
        pos1000: Math.round(posicion * 1000),
        cambiar: function (ev) {
          var t = Number(ev.target.value) / 1000, x = a + t * (b - a);
          var nuevo = log ? Math.exp(x) : x;
          if (r.paso) nuevo = Math.round(nuevo / r.paso) * r.paso;
          nuevo = log ? Number(nuevo.toPrecision(2)) : Number(nuevo.toFixed(4));
          var p = Object.assign({}, self.state.params); p[clave] = nuevo;
          self.setState({ params: p });
        },
        valorEstilo: 'position:absolute;top:0;left:calc(6px + (100% - 12px) * ' + posicion + ');transform:translateX(-50%);font-size:13px;color:#0f6f5c;pointer-events:none',
        relleno: 'position:absolute;left:6px;top:22px;height:4px;border-radius:2px;background:#0f6f5c;pointer-events:none;width:calc((100% - 12px) * ' + posicion + ')',
        pomo: 'position:absolute;top:18px;left:calc(6px + (100% - 12px) * ' + posicion + ');width:12px;height:12px;margin-left:-6px;border-radius:50%;background:#0f6f5c;pointer-events:none'
      });
    }
    function pistaBola(on) {
      return {
        pista: 'position:relative;width:32px;height:18px;border-radius:9px;flex:none;transition:background .15s ease;background:' + (on ? '#0f6f5c' : '#c9c8bf'),
        bola: 'position:absolute;top:2px;width:14px;height:14px;border-radius:50%;background:#fbfbf9;transition:left .15s ease;left:' + (on ? '16px' : '2px')
      };
    }
    function toggle(etiqueta, clave) {
      var on = !!fl[clave];
      return Object.assign(tipo('t'), pistaBola(on), {
        etiqueta: etiqueta, envoltorio: envoltorio(false), aria: on ? 'true' : 'false',
        pulsar: function () { var f = Object.assign({}, self.state.flags); f[clave] = !f[clave]; self.setState({ flags: f }); }
      });
    }
    function texto(id, clave, etiqueta, ayuda) {
      return Object.assign(tipo('x'), {
        id: id, etiqueta: etiqueta, valor: P[clave] != null ? String(P[clave]) : '', ayuda: ayuda, envoltorio: envoltorio(false),
        cambiar: function (ev) {
          var limpio = ev.target.value.replace(/[^0-9]/g, '').slice(0, 9);
          if (limpio !== ev.target.value) ev.target.value = limpio;
          var p = Object.assign({}, self.state.params); p[clave] = limpio === '' ? 0 : Number(limpio);
          self.setState({ params: p });
        }
      });
    }
    function seccion(id, titulo, resumen, mandos) {
      var abierta = !!st.abiertos[id];
      return {
        id: 'seccion-' + id, titulo: titulo, resumen: resumen, mandos: mandos, abierta: abierta,
        aria: abierta ? 'true' : 'false',
        chevron: 'flex:none;color:#767771;transition:transform .2s ease;transform:rotate(' + (abierta ? 90 : 0) + 'deg)',
        alternar: function () { var a = Object.assign({}, self.state.abiertos); a[id] = !a[id]; self.setState({ abiertos: a }); }
      };
    }
    var entero = function (v) { return String(Math.round(v)); };
    var dos = function (v) { return Number(v).toFixed(2); };
    var lr = function (v) { return String(Number(Number(v).toPrecision(2))); };
    var secciones = !st.params ? [] : [
      seccion('opt', 'Optimizacion', P.pasos + ' pasos · lr ' + lr(P.learning_rate) + ' · semilla ' + P.semilla, [
        slider('pasos', 'Pasos de optimizacion', entero, true),
        slider('learning_rate', 'Learning rate', lr, false),
        texto('semilla', 'semilla', 'Semilla', true)
      ]),
      seccion('frec', 'Penalizadores de frecuencia', 'L1 ' + dos(P.l1) + ' · variacion total ' + dos(P.tv) + ' · desenfoque ' + dos(P.blur), [
        slider('l1', 'L1', dos, false),
        slider('tv', 'Variacion total', dos, true),
        slider('blur', 'Desenfoque', dos, false)
      ]),
      seccion('rob', 'Robustez a transformaciones', fl.robustez ? 'jitter ' + P.jitter + ' px · rotacion ' + P.rotacion + '° · escala ' + dos(P.escala) : 'desactivada', [
        toggle('Activada', 'robustez'),
        slider('jitter', 'Jitter (px)', entero, false, !fl.robustez),
        slider('rotacion', 'Rotacion maxima (grados)', entero, false, !fl.robustez),
        slider('escala', 'Amplitud de escala', dos, false, !fl.robustez)
      ]),
      // La atenuacion solo existe en el espacio de Fourier: sin el, se apaga.
      seccion('pre', 'Preconditioning', fl.fourier ? 'Fourier decorrelado · atenuacion ' + dos(P.decay_power) : 'pixeles directos', [
        toggle('Espacio de Fourier decorrelado', 'fourier'),
        slider('decay_power', 'Atenuacion de frecuencias', dos, true, !fl.fourier)
      ])
    ];
    // W&B, en la barra superior.
    var tgW = pistaBola(!!fl.wandb);

    // La salida de la red: solo lectura, como en la v7 pero sobre fondo oscuro.
    var top5 = st.datos ? st.datos.top5.map(function (c) {
      var n = st.nombres && st.nombres[c[0]];
      return { binomio: n ? n[0] : 'clase ' + c[0], autor: n ? n[1] : '', p: c[1] };
    }) : [];
    var especies = top5.map(function (e, i) {
      return {
        binomio: e.binomio, autor: e.autor,
        pct: (e.p * 100).toFixed(1) + '%',
        peso: i === 0 ? 'font-weight:700;color:#edefe9' : 'color:#c3c8bd',
        barra: 'display:block;height:100%;border-radius:2px;width:' + (e.p * 100).toFixed(2) + '%;background:' + (i === 0 ? '#3ebd9d' : '#5f8f84')
      };
    });
    // El top-5 se apaga con el zoom en otra etapa; si la etapa es fc, sigue a la vista.
    var salidaEstilo = 'flex:0 1 400px;min-width:0;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;gap:16px;' +
      'padding:24px 28px;border-left:1px solid #2a2c28;transition:opacity .5s ease;opacity:' + (detalle && sel !== 4 ? 0.3 : 1);

    // ===================================================== dispositivo y optimizacion
    var T = st.trabajo;
    var fase = !T ? 'vacio' : T.estado === 'corriendo' ? 'generando' : 'hecho';
    var generando = fase === 'generando';
    var hayGpu = !!(st.info && st.info.gpu);
    function estiloDisp(activo, izq, apagado) {
      return 'display:flex;align-items:center;justify-content:center;gap:8px;min-height:36px;padding:0 14px;font:inherit;font-size:14px;cursor:' + (apagado ? 'not-allowed' : 'pointer') + ';' +
        'border:1px solid ' + (activo ? '#0f6f5c' : '#dddcd3') + ';' +
        (izq ? 'border-radius:6px 0 0 6px;' : 'border-radius:0 6px 6px 0;margin-left:-1px;') +
        'background:' + (activo ? '#e3efeb' : '#fbfbf9') + ';color:' + (activo ? '#0f6f5c' : '#1b1c19') + ';' +
        (activo ? 'font-weight:600;position:relative;z-index:1;' : '') + (apagado ? 'opacity:.45;' : '');
    }
    function elegir(d) {
      return function () {
        if (d === 'gpu' && !hayGpu) return;
        if (!self.state.trabajo || self.state.trabajo.estado !== 'corriendo') self.setState({ disp: d });
      };
    }
    var faltaObjetivo = obj === 'multi' && st.multi.canales.length === 0;
    var puedeGenerar = !!st.datos && !generando && !st.cargando && !faltaObjetivo;

    // ---- el mapa superpuesto a la foto: solo con Channel sobre una capa convolucional
    var conMapa = this.objetivoConMapa();
    var claveMapa = this.claveMapa();
    var mapaListo = conMapa && st.mapaImg && st.mapaClave === claveMapa;
    var mostrandoMapa = !!(mapa && mapaListo);
    var canalMapa = conMapa ? st.elegidos[sel] : null;
    var fotoUrl = st.foto ? '/api/fotos/' + st.foto.id + '/foto.png' : '';

    // ===================================================== metricas paso a paso
    // Las curvas reales del optimizador. El eje x va fijo a los pasos pedidos; el y sube
    // a saltos redondos solo cuando la curva lo supera, para que no se reescale a cada paso.
    var hist = T ? T.metricas : null;
    var NH = T ? T.pasos : (P.pasos || 256);
    var nVis = hist ? hist.activacion.length : 0;
    var cur = st.cursor != null && st.cursor < nVis ? st.cursor : null;
    function finitos(a) { return a.filter(function (v) { return v != null && isFinite(v); }); }
    function maxDe(a) { return finitos(a).reduce(function (m, v) { return v > m ? v : m; }, -Infinity); }
    function minDe(a) { return finitos(a).reduce(function (m, v) { return v < m ? v : m; }, Infinity); }
    function grafica(titulo, nota, serie, lo, hi, formato, ref) {
      serie = serie || [];
      var y = function (v) { return 136 - (v - lo) / (hi - lo) * 132; };
      var x = function (i) { return i / Math.max(1, NH - 1) * 400; };
      var pts = [];
      for (var i = 0; i < nVis && i < serie.length; i++) {
        if (serie[i] != null) pts.push(x(i).toFixed(1) + ',' + y(serie[i]).toFixed(1));
      }
      var foco = cur != null ? cur : nVis - 1;
      var hayPunto = nVis > 0 && foco >= 0 && serie[foco] != null;
      var px = hayPunto ? x(foco) / 4 : 0, py = hayPunto ? y(serie[foco]) / 1.4 : 0;
      var alinear = px < 15 ? '0' : px > 85 ? '-100%' : '-50%';
      var ultimo = nVis ? serie[nVis - 1] : null;
      return {
        titulo: titulo, nota: nota,
        valor: ultimo != null ? formato(ultimo) : '—',
        yMax: formato(hi), yMin: formato(lo),
        xMitad: 'paso ' + Math.round(NH / 2), xFin: String(NH),
        aria: titulo + ': ' + (nVis && ultimo != null ? 'de ' + formato(finitos(serie)[0]) + ' a ' + formato(ultimo) + ' en ' + nVis + ' pasos' : 'sin datos todavia'),
        marcas: [1, 2, 3].map(function (k) { return { x: (400 * k / 4).toFixed(0) }; }),
        puntos: pts.join(' '),
        hayRef: ref != null,
        refY: ref != null ? y(ref).toFixed(1) : '0',
        refEtiqueta: ref != null ? 'foto ' + formato(ref) : '',
        refEtiquetaEstilo: ref != null ? 'position:absolute;right:4px;top:calc(' + (y(ref) / 1.4).toFixed(1) + '% - 18px);font-size:11px;color:#767771;pointer-events:none' : '',
        puntoEstilo: hayPunto ? 'position:absolute;width:8px;height:8px;margin:-4px 0 0 -4px;border-radius:50%;background:#0f6f5c;box-shadow:0 0 0 2px #fbfbf9;pointer-events:none;left:' + px.toFixed(2) + '%;top:' + py.toFixed(2) + '%' : 'display:none',
        cursorEstilo: cur != null ? 'position:absolute;top:0;bottom:0;width:1px;background:#a3a99d;pointer-events:none;left:' + px.toFixed(2) + '%' : 'display:none',
        tooltip: cur != null && serie[cur] != null ? 'paso ' + (cur + 1) + ' · ' + formato(serie[cur]) : '',
        tooltipEstilo: cur != null && serie[cur] != null ? 'position:absolute;top:4px;left:' + px.toFixed(2) + '%;transform:translateX(' + alinear + ');padding:2px 6px;border-radius:4px;background:#1b1c19;color:#fbfbf9;font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums;pointer-events:none' : 'display:none'
      };
    }
    var f2 = function (v) { return (v >= 0 ? '+' : '') + v.toFixed(2); };
    var f3 = function (v) { return v.toFixed(3); };
    var f4 = function (v) { return v.toFixed(4); };
    // Que mide la grafica de activacion, segun el objetivo que se lanzo.
    function notaActivacion(o) {
      if (!o) return 'El valor del objetivo en cada paso. La linea gris es lo que lo activa la foto.';
      var foto = ' La linea gris es lo que lo activa la foto.';
      if (o.tipo === 'capa') return 'Media del cuadrado de toda ' + o.capa + ' (DeepDream).' + foto;
      if (o.tipo === 'multi') return 'Suma de ' + (o.capa === 'fc' ? 'los logits de las clases ' : 'las medias de los canales ') + o.canales.join(', ') + (o.capa === 'fc' ? '.' : ' de ' + o.capa + '.') + foto;
      if (o.tipo === 'neurona') return 'Canal ' + o.canal + ' de ' + o.capa + ' en la fila ' + o.x + ', columna ' + o.y + '.' + foto;
      if (o.capa === 'fc') return 'Logit de la clase ' + o.canal + (especie(o.canal) ? ' (' + st.nombres[o.canal][0] + ')' : '') + '.' + foto;
      return 'Media del canal ' + o.canal + ' de ' + o.capa + '.' + foto;
    }
    var vacias = { activacion: [], aspereza: [], cambio: [] };
    var H = hist || vacias;
    var ref = T && T.referencia != null ? T.referencia : null;
    var loA = Math.min(0, nVis ? minDe(H.activacion) : 0, ref != null ? ref : 0);
    loA = loA < 0 ? -topeRedondo(-loA) : 0;
    var hiA = topeRedondo(Math.max(nVis ? maxDe(H.activacion) : 0, ref != null ? ref : 0) * 1.05);
    var gA = grafica('Activacion', notaActivacion(T && T.objetivo), H.activacion, loA, hiA, f2, ref);
    var veces = nVis && ref ? H.activacion[nVis - 1] / ref : 0;
    if (nVis && ref > 0 && veces > 1) gA.valor = f2(H.activacion[nVis - 1]) + ' · ' + veces.toFixed(0) + '× la foto';
    var graficas = [
      gA,
      grafica('Aspereza', 'Variacion total de la imagen: sube a medida que aparece detalle fino.', H.aspereza, 0, topeRedondo((nVis ? maxDe(H.aspereza) : 0.1) * 1.05), f3, null),
      grafica('Cambio por paso', 'Cuanto se mueve la imagen de un paso al siguiente. Si se aplana, ha convergido.', H.cambio, 0, topeRedondo((nVis > 1 ? maxDe(H.cambio) : 0.01) * 1.05), f4, null)
    ];
    function moverGrafica(ev) {
      if (!nVis) return;
      var r = ev.currentTarget.getBoundingClientRect();
      var p = Math.round((ev.clientX - r.left) / r.width * (NH - 1));
      p = Math.max(0, Math.min(nVis - 1, p));
      if (p !== self.state.cursor) self.setState({ cursor: p });
    }
    function salirGrafica() { if (self.state.cursor != null) self.setState({ cursor: null }); }

    // ===================================================== la optimizacion, paso a paso
    // Los pasos se reparten entre N capturas: con 1000 pasos y N = 4, a los 250, 500, 750 y 1000.
    // La captura que esta en camino ensena la vista previa del paso actual.
    var N_CAPTURAS = st.info ? st.info.capturas : 4;
    var progreso = T ? T.paso / Math.max(1, T.pasos) : 0;
    var pasosCaptura = T && T.capturas_en.length ? T.capturas_en : [];
    var fotogramas = [];
    for (var kf = 0; kf < N_CAPTURAS; kf++) {
      (function (k) {
        var paso = pasosCaptura[k] || Math.round((T ? T.pasos : (P.pasos || 256)) * (k + 1) / N_CAPTURAS);
        var src = T ? T.capturas[k] : null;
        var hecho = !!src;
        var enCurso = !hecho && generando && T.capturas.length === k && !!T.vista;
        var final = k === N_CAPTURAS - 1;
        fotogramas.push({
          src: hecho ? src : enCurso ? T.vista : '', paso: 'paso ' + paso,
          alt: final ? 'Patron optimizado, paso ' + paso : 'Imagen a los ' + paso + ' pasos de optimizacion',
          hayImagen: hecho || enCurso,
          pendiente: !hecho && !enCurso,
          etiqueta: 'Paso ' + paso + (final ? ' · final' : ''),
          marco: 'position:relative;aspect-ratio:1;border-radius:4px;overflow:hidden;background:#f1f1ec;' +
            (!hecho && !enCurso ? 'border:1px dashed #c9c8bf;box-sizing:border-box;' : '') +
            (final && hecho ? 'outline:2px solid #0f6f5c;outline-offset:2px;' : ''),
          imgEstilo: 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;' + (enCurso ? 'opacity:.85' : ''),
          pieEstilo: 'display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:13px;' +
            (final && hecho ? 'color:#0f6f5c;font-weight:600' : hecho ? 'color:#1b1c19' : 'color:#767771')
        });
      })(kf);
    }
    // Marcas en la barra de progreso donde cae cada captura (menos la ultima, que es el final).
    var marcas = [];
    for (var m = 1; m < N_CAPTURAS; m++) {
      marcas.push({ estilo: 'position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:#fbfbf9;left:' + (100 * m / N_CAPTURAS) + '%' });
    }
    var tg = pistaBola(mapa && conMapa);
    var foto = st.foto;
    var kb = foto ? foto.bytes / 1024 : 0;
    return {
      etiquetaModelo: MOD.etiqueta,
      wandbUrl: st.info ? st.info.wandb : '#',
      // la foto
      zonaClase: 'zona-subida' + (st.arrastrando ? ' arrastrando' : ''),
      arrastrar: function (ev) { ev.preventDefault(); if (!self.state.arrastrando) self.setState({ arrastrando: true }); },
      dejarArrastrar: function () { self.setState({ arrastrando: false }); },
      soltar: function (ev) { ev.preventDefault(); self.setState({ arrastrando: false }); self.subir(ev.dataTransfer.files[0]); },
      abrirFichero: function () { document.getElementById('subida').click(); },
      elegirFichero: function (ev) { self.subir(ev.target.files[0]); ev.target.value = ''; },
      fotoNombre: foto ? foto.nombre : 'Cargando…',
      fotoTamano: st.errorFoto ? st.errorFoto : foto ? (kb >= 1024 ? (kb / 1024).toFixed(1) + 'MB' : kb.toFixed(1) + 'KB') : '',
      fotoTamanoEstilo: 'font-size:12px;color:' + (st.errorFoto ? '#a3412c' : '#767771'),
      quitarFoto: function () { self.elegirMuestra(''); },
      muestras: st.info ? st.info.muestras : [],
      muestraSel: st.muestraSel,
      elegirMuestra: function (ev) { self.elegirMuestra(ev.target.value); },

      objetivos: objetivos,
      entrada: entrada, bloques: bloques, flechas: flechas,
      camaraEstilo: camaraEstilo, veloEstilo: veloEstilo,
      flechasEstilo: 'position:absolute;left:0;top:0;transition:opacity .4s ease;opacity:' + (detalle ? 0 : 1),
      general: !detalle, enCanales: detalle && !nivel2, enMapa: nivel2,
      pista: PISTAS[obj],
      volver: function () { self.setState({ detalle: false, nivel2: false }); },
      volverCanales: function () { self.setState({ nivel2: false, hover: null }); },
      resumen: st.datos ? resumen : '',
      resumenEstilo: 'position:absolute;left:20px;bottom:16px;margin:0;font-family:\'Source Code Pro\',ui-monospace,monospace;font-size:12px;color:#c3c8bd;transition:opacity .4s ease;opacity:' + (detalle ? 0 : 1),
      panelOculto: abierto1 ? 'false' : 'true',
      panelEstilo: panelEstilo, panelNombre: bS.capa, panelMeta: panelMeta, panelInfo: panelInfo,
      hayEspecie: !!nombreFc, especieBinomio: nombreFc ? nombreFc[0] : '', especieAutor: nombreFc ? nombreFc[1] : '',
      rejillaEstilo: rejillaEstilo, celdas: celdas, leyendaMin: leyendaMin, leyendaMax: leyendaMax,
      hayLimpiar: obj === 'multi' && marcados.length > 0,
      hayTira: hayTira, tira: tira, tiraTexto: tiraTexto, tiraEstilo: tiraEstilo,
      limpiar: function () { self.setState({ multi: { etapa: self.state.sel, canales: [] } }); },
      mapaOculto: nivel2 ? 'false' : 'true',
      mapaPanelEstilo: mapaPanelEstilo, mapaEstilo: mapaEstilo, mapaClase: mapaClase,
      mapaPos: mapaPos, mapaMeta: mapaMeta, mapaCanal: mapaCanal,
      marcaEstilo: marcaEstilo, hoverEstilo: hoverEstilo, selectorAria: selectorAria,
      campoEstilo: campoEstilo, campoTexto: campoTexto, campoAria: campoAria,
      selectorTab: nivel2 ? 0 : -1,
      pulsarMapa: pulsarMapa, moverMapa: moverMapa, salirMapa: salirMapa, teclaMapa: teclaMapa,

      secciones: secciones, especies: especies,
      modelos: Object.keys(st.info ? st.info.modelos : { resnet18: 1, resnet50: 1, resnet152: 1 }).map(function (a, k, todos) {
        var activo = a === st.modelo;
        var disponible = !st.info || st.info.modelos[a].disponible;
        var bloqueado = !disponible || (generando && !activo);
        return {
          etiqueta: st.info ? st.info.modelos[a].etiqueta : a, aria: activo ? 'true' : 'false',
          inhabilitado: bloqueado,
          titulo: !disponible ? 'Falta el checkpoint: python scripts/get_data.py --pesos ' + a :
            bloqueado ? 'Espera a que termine la optimizacion' : '',
          pulsar: function () { if (a !== self.state.modelo && !bloqueado) self.cargarModelo(a); },
          estilo: 'font:inherit;font-size:13px;min-height:30px;padding:0 12px;cursor:' + (bloqueado ? 'not-allowed' : 'pointer') + ';' +
            (k === 0 ? 'border-radius:5px 0 0 5px;' : k === todos.length - 1 ? 'border-radius:0 5px 5px 0;margin-left:-1px;' : 'border-radius:0;margin-left:-1px;') +
            'border:1px solid ' + (activo ? '#3ebd9d' : '#3a3f39') + ';background:' + (activo ? 'rgba(62,189,157,.14)' : 'transparent') +
            ';color:' + (activo ? '#edefe9' : '#a3a99d') + (activo ? ';font-weight:600;position:relative;z-index:1' : '') +
            (bloqueado ? ';opacity:.4' : '')
        };
      }),
      selectorEstilo: 'position:absolute;right:16px;top:10px;display:flex;transition:opacity .4s ease;' +
        (detalle ? 'opacity:0;pointer-events:none' : 'opacity:1'),
      cargandoTexto: st.cargando ? 'Cargando ' + MOD.etiqueta + '…' : '',
      cargandoEstilo: 'position:absolute;left:0;right:0;top:176px;margin:0;text-align:center;font-size:14px;color:#edefe9;pointer-events:none;transition:opacity .3s ease;opacity:' + (st.cargando ? 1 : 0),
      wandbAria: fl.wandb ? 'true' : 'false', wandbPista: tgW.pista, wandbBola: tgW.bola,
      alternarWandb: function () { var f = Object.assign({}, self.state.flags); f.wandb = !f.wandb; self.setState({ flags: f }); },
      salidaEstilo: salidaEstilo,
      // la foto con el mapa
      mapa: mostrandoMapa,
      mapaAria: mapa && conMapa ? 'true' : 'false',
      mapaNo: !conMapa,
      mapaTitulo: conMapa ? '' : 'Solo con el objetivo Channel sobre una capa convolucional: un canal en toda la foto',
      mapaBotonEstilo: 'display:flex;align-items:center;gap:10px;min-height:28px;padding:0;border:none;background:none;font:inherit;font-size:14px;color:#1b1c19;text-align:left;' +
        (conMapa ? 'cursor:pointer' : 'cursor:not-allowed;opacity:.45'),
      pistaMapa: tg.pista, bolaMapa: tg.bola,
      alternarMapa: function () { if (self.objetivoConMapa()) self.setState({ mapa: !self.state.mapa }); },
      imagenIzq: mostrandoMapa ? st.mapaImg : fotoUrl,
      altIzq: mostrandoMapa ? 'Foto de la planta con el mapa de activacion del canal ' + canalMapa + ' superpuesto' : 'Foto de la planta a analizar',
      textoMapa: conMapa ? 'Mapa del canal ' + canalMapa : 'Mapa de activacion',
      escalaMax: '+' + Number(st.mapaMax || 0).toFixed(2),
      escalaAria: 'Escala del mapa: activacion del canal ' + canalMapa + ' de 0 a +' + Number(st.mapaMax || 0).toFixed(2),
      // dispositivo
      cpuAria: st.disp === 'cpu' ? 'true' : 'false',
      gpuAria: st.disp === 'gpu' ? 'true' : 'false',
      cpuEstilo: estiloDisp(st.disp === 'cpu', true, false),
      gpuEstilo: estiloDisp(st.disp === 'gpu', false, !hayGpu),
      gpuNo: !hayGpu,
      gpuTitulo: hayGpu ? '' : 'No hay GPU disponible: torch no ve ninguna',
      elegirCpu: elegir('cpu'), elegirGpu: elegir('gpu'),
      // optimizacion
      fotogramas: fotogramas, marcas: marcas,
      graficas: graficas, moverGrafica: moverGrafica, salirGrafica: salirGrafica,
      generar: function () { if (puedeGenerar) self.generar(); },
      generando: generando || !puedeGenerar,
      textoBoton: generando ? 'Generando...' : 'Generar',
      botonEstilo: 'display:flex;align-items:center;justify-content:center;gap:8px;min-height:44px;border:none;border-radius:6px;font:inherit;font-size:16px;color:#ffffff;' +
        (generando ? 'background:#5c9a8c;cursor:progress' : puedeGenerar ? 'background:#0f6f5c;cursor:pointer' : 'background:#9fb8b1;cursor:not-allowed'),
      errorTexto: st.errorTexto || (faltaObjetivo && !generando ? 'Marca al menos un canal para optimizar.' : ''),
      progresoNum: Math.round(progreso * 100),
      progresoEstilo: 'height:100%;border-radius:3px;background-color:#5f8f84;width:' + (progreso * 100).toFixed(1) + '%;' +
        (generando ? 'background-image:linear-gradient(45deg,rgba(255,255,255,.16) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.16) 50%,rgba(255,255,255,.16) 75%,transparent 75%,transparent);background-size:20px 20px;animation:avance .8s linear infinite' : ''),
      tiempo: fase === 'vacio' ? '—' : T.segundos.toFixed(1) + ' s',
      tiempoColor: fase === 'vacio' ? 'color:#767771' : 'color:#1b1c19;font-weight:600'
    };
  }
}

render(html`<${App} />`, document.getElementById('app'));
