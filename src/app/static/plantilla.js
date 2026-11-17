// Generado a partir de la plantilla <x-dc> del diseno (V13.dc.html) y retocado a mano solo
// donde un control tenia que funcionar de verdad: subida de fotos, muestras, deslizadores,
// semilla, GPU, modelos sin checkpoint, mapa superpuesto bloqueable y la linea de error.
import { h } from './vendor/preact.mjs';
import htm from './vendor/htm.mjs';

const html = htm.bind(h);

export function plantilla(v) {
  return html`<div style="min-height: 2340px; display: flex; background: #fbfbf9; color: #1b1c19; font-family: 'Source Sans 3', 'Source Sans Pro', system-ui, sans-serif; font-size: 16px; line-height: 1.5">


  
  <main style="flex: 999 1 560px; min-width: 0; box-sizing: border-box; padding: 24px 80px 96px; display: flex; flex-direction: column; gap: 16px">
    <div style="display: flex; justify-content: space-between; gap: 12px; align-items: center; min-height: 40px; color: #1b1c19">
      <div role="group" aria-label="Dispositivo de calculo" style="display: flex; align-items: center; gap: 12px">
        <span style="font-size: 14px; color: #5f605a">Dispositivo</span>
        <div style="display: inline-flex">
          <button type="button" aria-pressed=${v.cpuAria} onClick=${v.elegirCpu} style=${v.cpuEstilo}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1.5"></rect><rect x="9.5" y="9.5" width="5" height="5" rx="0.5"></rect><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"></path></svg>
            CPU
          </button>
          <button type="button" aria-pressed=${v.gpuAria} onClick=${v.elegirGpu} style=${v.gpuEstilo} disabled=${v.gpuNo} title=${v.gpuTitulo}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="6" width="20" height="11" rx="1.5"></rect><circle cx="8.5" cy="11.5" r="3"></circle><circle cx="8.5" cy="11.5" r="0.6"></circle><path d="M14.5 9.5h4M14.5 12h4M14.5 14.5h4"></path><path d="M5 17v3M8 17v3M11 17v3"></path></svg>
            GPU
          </button>
        </div>
      </div>
      
      <div role="group" aria-label="Weights & Biases" style="display: flex; align-items: center; gap: 16px; margin-left: auto; min-height: 32px">
        <button type="button" role="switch" aria-checked=${v.wandbAria} onClick=${v.alternarWandb} style="display: flex; align-items: center; gap: 10px; min-height: 36px; padding: 0; border: none; background: none; font: inherit; font-size: 14px; color: #1b1c19; cursor: pointer">
          <span style=${v.wandbPista}><span style=${v.wandbBola}></span></span>
          <span>Registrar en W&B</span>
        </button>
        <a href=${v.wandbUrl} target="_blank" rel="noopener" style="display: inline-flex; align-items: center; gap: 6px; min-height: 36px; padding: 0 12px; border: 1px solid #dddcd3; border-radius: 6px; background: #fbfbf9; color: #1b1c19; text-decoration: none; font-size: 14px">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18"></path><path d="M7 15l4-4 3 3 5-6"></path></svg>
          Ver los runs
        </a>
      </div>
    </div>

    <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 16px">
      <h1 style="margin: 0; font-size: 44px; font-weight: 700; line-height: 1.2; letter-spacing: -0.01em">Visualizacion de caracteristicas</h1>
      <p style="margin: 0; font-size: 14px; color: #767771">Olah, Mordvintsev & Schubert (2017) sobre un ${v.etiquetaModelo} preentrenado en Pl@ntNet-300K. Los pesos van congelados: lo que se optimiza es la imagen.</p>
    </div>

    <div style="display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end">
      <div style="flex: 1 1 320px; min-width: 0; display: flex; flex-direction: column; gap: 4px">
        <label for="subida" style="font-size: 14px">Sube una foto de una planta</label>
        <div class=${v.zonaClase} onDragOver=${v.arrastrar} onDragLeave=${v.dejarArrastrar} onDrop=${v.soltar} style="display: flex; align-items: center; gap: 16px; padding: 16px; border-radius: 6px; background: #f1f1ec">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#767771" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 18a5 5 0 0 1-.5-9.97A6 6 0 0 1 18 9a4 4 0 0 1 0 9"></path><path d="M12 12v8"></path><path d="M9 15l3-3 3 3"></path></svg>
          <div style="flex: 1; min-width: 0; display: flex; flex-direction: column">
            <span style="font-size: 15px">Drag and drop file here</span>
            <span style="font-size: 13px; color: #767771">Limit 200MB per file • JPG, JPEG, PNG</span>
          </div>
          <input id="subida" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" style="display: none" onChange=${v.elegirFichero} />
          <button type="button" onClick=${v.abrirFichero} style="font: inherit; font-size: 15px; padding: 6px 14px; border: 1px solid #dddcd3; border-radius: 6px; background: #fbfbf9; color: #1b1c19; cursor: pointer">Browse files</button>
        </div>
        <div style="display: flex; align-items: center; gap: 10px; padding: 6px 4px">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#767771" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"></path><path d="M14 3v5h5"></path></svg>
          <span style="font-size: 14px; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">${v.fotoNombre}</span>
          <span style=${v.fotoTamanoEstilo}>${v.fotoTamano}</span>
          <span style="flex: 1"></span>
          <button type="button" aria-label=${"Quitar " + v.fotoNombre} onClick=${v.quitarFoto} style="border: none; background: none; padding: 4px; color: #767771; cursor: pointer">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12"></path><path d="M18 6L6 18"></path></svg>
          </button>
        </div>
      </div>
      <div style="flex: 1 1 320px; min-width: 0; display: flex; flex-direction: column; gap: 4px; padding-bottom: 46px">
        <label for="muestra" style="display: flex; align-items: center; gap: 6px; font-size: 14px">O elige una del conjunto de test
          <span title=${v.ayudaMuestra} aria-label=${v.ayudaMuestra} role="img" tabindex="0" style="display: inline-flex; cursor: help"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#767771" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg></span>
        </label>
        <select id="muestra" value=${v.muestraSel} onChange=${v.elegirMuestra} style="font: inherit; font-size: 15px; height: 40px; padding: 0 12px; border: 1px solid #dddcd3; border-radius: 6px; background: #f1f1ec; color: #1b1c19">
          <option value="">(ninguna)</option>
          ${(v.muestras || []).map((m) => html`<option value=${m.nombre}>${m.etiqueta}</option>`)}
        </select>
      </div>
    </div>

    
    <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 16px">
      <h2 style="margin: 0; font-size: 28px; font-weight: 600; line-height: 1.3">La red, por etapas</h2>
      <p style="margin: 0; font-size: 14px; color: #767771">Elige que quieres optimizar y despues donde: una etapa entera, uno o varios canales, o una sola neurona de un canal.</p>
    </div>

    
    <fieldset style="border: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px">
      <legend style="display: flex; align-items: center; gap: 6px; font-size: 14px; padding: 0; margin-bottom: 4px">Objetivo
        <span title=${v.ayudaObjetivo} aria-label=${v.ayudaObjetivo} role="img" tabindex="0" style="display: inline-flex; cursor: help"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#767771" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg></span>
      </legend>
      <div style="display: inline-flex; flex-wrap: wrap; width: fit-content">
        ${(v.objetivos || []).map((o) => html`
          <button type="button" aria-pressed=${o.aria} onClick=${o.pulsar} style=${o.estilo}>${o.etiqueta}</button>
        `)}
      </div>
    </fieldset>

    
    <div style="display: flex; background: #141513; border-radius: 10px; overflow: hidden">
      
      <div style="position: relative; flex: 1 0 auto; min-width: 0; display: flex; justify-content: center; padding: 8px 16px; overflow: hidden">
        <div style="position: relative; flex: none; width: 880px; height: 360px; color: #edefe9">

          <div class="camara" style=${v.camaraEstilo}>
            <svg width="880" height="360" viewBox="0 0 880 360" style=${v.flechasEstilo} aria-hidden="true">
              <defs><marker id="punta" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="#7c8277"></path></marker></defs>
              ${(v.flechas || []).map((f) => html`
                <line x1=${f.x1} y1="175" x2=${f.x2} y2="175" stroke="#7c8277" stroke-width="1.5" marker-end="url(#punta)"></line>
              `)}
            </svg>

            <div style=${v.entrada.posEstilo}>
              <div style=${v.entrada.cuboEstilo}>
                ${(v.entrada.caras || []).map((c) => html`
                  <div class=${c.clase} style=${c.estilo}></div>
                `)}
              </div>
            </div>
            <div style=${v.entrada.rotuloEstilo}>
              <span style="font-weight: 600">Entrada</span>
              <span style="font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 11px; color: #a3a99d">3×224×224</span>
            </div>

            ${(v.bloques || []).map((b) => html`
              <button type="button" class="etapa" onClick=${b.pulsar} aria-label=${b.aria} aria-pressed=${b.elegidaAria} aria-disabled=${b.inhabilitadaAria} style=${b.posEstilo}>
                <span style=${b.cuboEstilo}>
                  ${(b.caras || []).map((c) => html`
                    <span class=${c.clase} style=${c.estilo}></span>
                  `)}
                </span>
              </button>
              <div style=${b.rotuloEstilo}>
                <span style=${b.nombreEstilo}>${b.nombre}</span>
                <span style="font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 11px; color: #a3a99d">${b.forma}</span>
              <span style="font-size: 11px; color: #7c8277">${b.bloques}</span>
              </div>
            `)}
          </div>

          
          <div class="velo" style=${v.veloEstilo}></div>


          
          <div class="panel-canales" aria-hidden=${v.panelOculto} style=${v.panelEstilo}>
            <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap">
              <span style="font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 15px; font-weight: 600; color: #edefe9">${v.panelNombre}</span>
              ${(v.hayEspecie) ? html`
                <span style="font-size: 15px; color: #edefe9"><em>${v.especieBinomio}</em> <span style="color: #c3c8bd">${v.especieAutor}</span></span>
                ${(v.especieProb) ? html`
                  <span style="font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 15px; font-weight: 600; color: #3ebd9d">${v.especieProb}</span>
                ` : null}
              ` : null}
              <!-- Renglon propio siempre: con el nombre de la especie y su probabilidad
                   delante, que miden lo que midan, esta linea caia unas veces al lado y
                   otras debajo, y la cabecera bailaba al cambiar de clase. -->
              <span style="flex: 0 0 100%; font-size: 13px; color: #a3a99d">${v.panelMeta}</span>
            </div>
            
            ${(v.hayTira) ? html`
              <div role="group" aria-label="Bloque de la etapa" style="display: flex; align-items: center; gap: 10px">
                <span style="flex: none; font-size: 12px; color: #a3a99d">Bloque</span>
                <div style=${v.tiraEstilo}>
                  ${(v.tira || []).map((t) => html`
                    <button type="button" class="bloque" title=${t.titulo} aria-label=${t.titulo} aria-pressed=${t.pulsado} onClick=${t.pulsar} style=${t.estilo}></button>
                  `)}
                </div>
                <span style="flex: none; min-width: 52px; text-align: right; font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 12px; color: #edefe9">${v.tiraTexto}</span>
              </div>
            ` : null}
            <div style=${v.rejillaEstilo}>
              ${(v.celdas || []).map((c) => html`
                <button type="button" class="celda" title=${c.titulo} aria-label=${c.titulo} aria-pressed=${c.aria} onClick=${c.pulsar} style=${c.estilo}></button>
              `)}
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; font-size: 13px; color: #c3c8bd; min-height: 28px">
              <div style="display: flex; align-items: center; gap: 8px; font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 11px; color: #a3a99d">
                <span>${v.leyendaMin}</span>
                <span style="width: 120px; height: 8px; border-radius: 2px; background: linear-gradient(to right, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4)"></span>
                <span>${v.leyendaMax}</span>
              </div>
              <div style="display: flex; align-items: center; gap: 12px">
                <span>${v.panelInfo}</span>
                ${(v.hayLimpiar) ? html`
                  <button type="button" class="limpiar" onClick=${v.limpiar} style="min-height: 28px; padding: 0 10px; border: 1px solid #3a3f39; border-radius: 4px; background: none; font: inherit; font-size: 13px; color: #c3c8bd; cursor: pointer">Limpiar</button>
                ` : null}
              </div>
            </div>
          </div>

          
          <div class="panel-mapa" aria-hidden=${v.mapaOculto} style=${v.mapaPanelEstilo}>
            <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap">
              <span style="font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 15px; font-weight: 600; color: #edefe9">${v.panelNombre} · canal ${v.mapaCanal}</span>
              <span style="font-size: 13px; color: #a3a99d">${v.mapaMeta}</span>
            </div>
            <div style="display: flex; gap: 24px; align-items: flex-start">
              <div style="position: relative; flex: none; width: 252px; height: 252px; outline: 1px solid rgba(237,239,233,.25)">
                <div class=${v.mapaClase} style=${v.mapaEstilo}></div>
                <div style=${v.hoverEstilo}></div>
                <div style=${v.marcaEstilo}></div>
                <button type="button" class="selector-pixel" aria-label=${v.selectorAria} tabIndex=${v.selectorTab} onClick=${v.pulsarMapa} onMouseMove=${v.moverMapa} onMouseLeave=${v.salirMapa} onKeyDown=${v.teclaMapa} style="position: absolute; inset: 0; z-index: 3; border: none; padding: 0; margin: 0; background: transparent"></button>
              </div>
              <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px; color: #c3c8bd; width: 200px">
                <span style="font-size: 15px; color: #edefe9; font-weight: 600">${v.mapaPos}</span>
                
                <div role="img" aria-label=${v.campoAria} class="hoja-foto" style="position: relative; width: 160px; height: 160px; border-radius: 4px; overflow: hidden; background-size: cover; background-position: center">
                  <span style=${v.campoEstilo}></span>
                </div>
                <span>${v.campoTexto}</span>
              </div>
            </div>
          </div>
        </div>

        
          
        <div role="group" aria-label="Modelo" style=${v.selectorEstilo}>
          ${(v.modelos || []).map((m) => html`
            <button type="button" class="modelo" aria-pressed=${m.aria} onClick=${m.pulsar} style=${m.estilo} disabled=${m.inhabilitado} title=${m.titulo}>${m.etiqueta}</button>
          `)}
        </div>
        <p aria-live="polite" style=${v.cargandoEstilo}>${v.cargandoTexto}</p>

        
          ${(v.general) ? html`
            <p style="position: absolute; left: 20px; top: 14px; margin: 0; font-size: 13px; color: #a3a99d">${v.pista}</p>
          ` : null}
          ${(v.enCanales) ? html`
            <button type="button" class="volver" onClick=${v.volver} style="position: absolute; left: 12px; top: 8px; display: flex; align-items: center; gap: 6px; min-height: 36px; padding: 0 10px; border: none; background: none; font: inherit; font-size: 14px; color: #c3c8bd; cursor: pointer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 18l-6-6 6-6"></path></svg>
              Todas las capas
            </button>
          ` : null}
          ${(v.enMapa) ? html`
            <button type="button" class="volver" onClick=${v.volverCanales} style="position: absolute; left: 12px; top: 8px; display: flex; align-items: center; gap: 6px; min-height: 36px; padding: 0 10px; border: none; background: none; font: inherit; font-size: 14px; color: #c3c8bd; cursor: pointer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 18l-6-6 6-6"></path></svg>
              Canales
            </button>
          ` : null}

          
          <p aria-live="polite" style=${v.resumenEstilo}>${v.resumen}</p>
      </div>
      <section aria-labelledby="salida" style=${v.salidaEstilo}>
        <h3 id="salida" style="margin: 0; font-size: 16px; font-weight: 600; color: #edefe9">Especies mas probables</h3>
        ${(v.especies || []).map((e) => html`
          <div style="display: flex; flex-direction: column; gap: 4px">
            <span style="display: flex; justify-content: space-between; align-items: baseline; gap: 12px; font-size: 14px">
              <span style=${`min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; ${e.peso}`}><em>${e.binomio}</em> ${e.autor}</span>
              <span style=${`flex: none; font-variant-numeric: tabular-nums; ${e.peso}`}>${e.pct}</span>
            </span>
            <span style="display: block; height: 10px; border-radius: 2px; background: #2a2c28">
              <span style=${e.barra}></span>
            </span>
          </div>
        `)}
      </section>
    </div>

    
    <section aria-labelledby="parametros" style="display: flex; flex-direction: column; gap: 8px; margin-top: 24px">
      <h2 id="parametros" style="margin: 0 0 8px; font-size: 28px; font-weight: 600; line-height: 1.3">Parametros de la optimizacion</h2>
      <div style="border-top: 1px solid #dddcd3">
        ${(v.secciones || []).map((s) => html`
          <div style="border-bottom: 1px solid #dddcd3">
            <button type="button" class="desplegable" aria-expanded=${s.aria} aria-controls=${s.id} onClick=${s.alternar} style="display: flex; align-items: center; gap: 12px; width: 100%; min-height: 52px; padding: 0 8px; border: none; background: none; font: inherit; color: #1b1c19; cursor: pointer; text-align: left">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style=${s.chevron}><path d="M9 6l6 6-6 6"></path></svg>
              <span style="font-size: 16px; font-weight: 600">${s.titulo}</span>
              <span style="flex: 1"></span>
              <span style="font-size: 14px; color: #767771; font-variant-numeric: tabular-nums">${s.resumen}</span>
            </button>
            ${(s.abierta) ? html`
              <div id=${s.id} style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px 48px; align-items: start; padding: 4px 8px 28px 38px">
                ${(s.mandos || []).map((m) => html`
                  <div style=${m.envoltorio}>
                    ${(m.esToggle) ? html`
                      <button type="button" role="switch" aria-checked=${m.aria} onClick=${m.pulsar} style="display: flex; align-items: center; gap: 10px; min-height: 40px; margin-top: 25px; padding: 0; border: none; background: none; font: inherit; font-size: 14px; color: #1b1c19; cursor: pointer; text-align: left">
                        <span style=${m.pista}><span style=${m.bola}></span></span>
                        <span>${m.etiqueta}</span>
                      </button>
                    ` : null}
                    ${(m.esTexto) ? html`
                      <label for=${m.id} style="display: flex; align-items: center; gap: 6px; font-size: 14px">${m.etiqueta}
                        ${(m.ayuda) ? html`
                          <span title=${m.ayuda} aria-label=${m.ayuda} role="img" tabindex="0" style="display: inline-flex; cursor: help"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#767771" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg></span>
                        ` : null}
                      </label>
                      <input id=${m.id} type="text" inputmode="numeric" autocomplete="off" value=${m.valor} onInput=${m.cambiar} style="font: inherit; font-size: 15px; height: 40px; box-sizing: border-box; padding: 0 12px; border: 1px solid #dddcd3; border-radius: 6px; background: #fbfbf9; color: #1b1c19; font-variant-numeric: tabular-nums" />
                    ` : null}
                    ${(m.esSlider) ? html`
                      <div style="display: flex; align-items: center; gap: 6px; font-size: 14px">
                        <span>${m.etiqueta}</span>
                        ${(m.ayuda) ? html`
                          <span title=${m.ayuda} aria-label=${m.ayuda} role="img" tabindex="0" style="display: inline-flex; cursor: help"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#767771" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg></span>
                        ` : null}
                      </div>
                      <div style="position: relative; height: 40px">
                        <span style=${m.valorEstilo}>${m.valor}</span>
                        <div style="position: absolute; left: 6px; right: 6px; top: 22px; height: 4px; border-radius: 2px; background: #dddcd3"></div>
                        <div style=${m.relleno}></div>
                        <input type="range" class="deslizador" min="0" max="1000" step="1" value=${m.pos1000} onInput=${m.cambiar} aria-label=${m.etiqueta} aria-valuetext=${m.valor} />
                        <div class="foco-deslizador" style=${m.pomo}></div>
                      </div>
                      <div style="display: flex; justify-content: space-between; font-size: 12px; color: #767771; margin-top: -8px">
                        <span>${m.min}</span><span>${m.max}</span>
                      </div>
                    ` : null}
                    ${(m.esEnlace) ? html`
                      <a href=${m.url} style="display: inline-flex; align-items: center; gap: 8px; width: fit-content; min-height: 40px; margin-top: 25px; padding: 0 14px; border: 1px solid #dddcd3; border-radius: 6px; background: #fbfbf9; color: #1b1c19; text-decoration: none; font-size: 15px">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18"></path><path d="M7 15l4-4 3 3 5-6"></path></svg>
                        ${m.etiqueta}
                      </a>
                    ` : null}
                  </div>
                `)}
              </div>
            ` : null}
          </div>
        `)}
      </div>
    </section>


    <hr style="border: none; border-top: 1px solid #dddcd3; margin: 16px 0; width: 100%" />

    
    
    <div style="display: grid; grid-template-columns: calc((100% - 84px) / 5) minmax(0, 1fr); column-gap: 48px; align-items: start">
      <section style="display: flex; flex-direction: column; gap: 12px; min-width: 0">
        <h2 style="margin: 0; font-size: 28px; font-weight: 600; line-height: 1.3">Que ve el modelo</h2>
        <button type="button" role="switch" aria-checked=${v.mapaAria} onClick=${v.alternarMapa} disabled=${v.mapaNo} title=${v.mapaTitulo} style=${v.mapaBotonEstilo}>
          <span style=${v.pistaMapa}><span style=${v.bolaMapa}></span></span>
          <span style="white-space: nowrap">${v.textoMapa}</span>
        </button>
        <div style="position: relative; width: 100%">
          <img src=${v.imagenIzq} alt=${v.altIzq} style="width: 100%; aspect-ratio: 1; object-fit: cover; display: block; border-radius: 4px" />
          ${(v.mapa) ? html`
            <div role="img" aria-label=${v.escalaAria} style="position: absolute; top: 0; bottom: 0; right: calc(100% + 14px); width: 52px; display: flex; flex-direction: column; align-items: flex-end; gap: 6px; font-size: 12px; color: #5f605a; font-variant-numeric: tabular-nums">
              <span>${v.escalaMax}</span>
              <div style="flex: 1; width: 12px; border-radius: 2px; background: linear-gradient(to top, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4)"></div>
              <span>0</span>
            </div>
          ` : null}
        </div>
      </section>
      <section style="display: flex; flex-direction: column; gap: 12px; min-width: 0">
        
        <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 16px">
          <h2 style="margin: 0; font-size: 28px; font-weight: 600; line-height: 1.3">Que patron busca</h2>
          <span aria-live="polite" style=${`flex: none; font-size: 14px; font-variant-numeric: tabular-nums; ${v.tiempoColor}`}>${v.tiempo}</span>
        </div>
        <div style="display: flex; align-items: center; min-height: 28px">
          <div role="progressbar" aria-label="Progreso de la optimizacion" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${v.progresoNum} style="position: relative; flex: 1; height: 14px; border-radius: 3px; background: #ecebe5; overflow: hidden">
            <div class="progreso-relleno" style=${v.progresoEstilo}></div>
            ${(v.marcas || []).map((k) => html`
              <span style=${k.estilo}></span>
            `)}
          </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px">
          ${(v.fotogramas || []).map((f) => html`
            <figure style="margin: 0; display: flex; flex-direction: column; gap: 6px; min-width: 0">
              <div style=${f.marco}>
                ${(f.hayImagen) ? html`
                  <img src=${f.src} alt=${f.alt} style=${f.imgEstilo} />
                ` : null}
                ${(f.pendiente) ? html`
                  <span style="position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 13px; color: #767771; font-variant-numeric: tabular-nums">${f.paso}</span>
                ` : null}
              </div>
              <figcaption style=${f.pieEstilo}>
                <span>${f.etiqueta}</span>
              </figcaption>
            </figure>
          `)}
        </div>
        <button type="button" onClick=${v.generar} disabled=${v.generando} style=${v.botonEstilo}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"></path><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"></path></svg>
          ${v.textoBoton}
        </button>
        <p aria-live="polite" style="margin: 0; min-height: 20px; font-size: 13px; line-height: 20px; color: #a3412c">${v.errorTexto}</p>
      </section>
    </div>

    
    <section aria-labelledby="metricas" style="display: flex; flex-direction: column; gap: 16px; margin-top: 16px">
      <h2 id="metricas" style="margin: 0; font-size: 28px; font-weight: 600; line-height: 1.3">Durante la optimizacion</h2>
      <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 40px">
        ${(v.graficas || []).map((g) => html`
          <figure style="margin: 0; display: flex; flex-direction: column; gap: 2px; min-width: 0">
            <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 12px">
              <figcaption style="font-size: 16px; font-weight: 600; color: #1b1c19">${g.titulo}</figcaption>
              <span aria-live="off" style="font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; color: #1b1c19; white-space: nowrap">${g.valor}</span>
            </div>
            <span style="font-size: 13px; color: #767771; min-height: 40px">${g.nota}</span>
            <div style="display: grid; grid-template-columns: 44px minmax(0, 1fr); column-gap: 8px; margin-top: 8px">
              <div aria-hidden="true" style="position: relative; height: 140px; font-size: 12px; color: #767771; font-variant-numeric: tabular-nums">
                <span style="position: absolute; right: 0; top: -8px">${g.yMax}</span>
                <span style="position: absolute; right: 0; bottom: -8px">${g.yMin}</span>
              </div>
              <div role="img" aria-label=${g.aria} style="position: relative; height: 140px">
                <svg viewBox="0 0 400 140" preserveAspectRatio="none" style="position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible" aria-hidden="true">
                  <line x1="0" y1="0" x2="400" y2="0" stroke="#ecebe5" stroke-width="1" vector-effect="non-scaling-stroke"></line>
                  <line x1="0" y1="70" x2="400" y2="70" stroke="#ecebe5" stroke-width="1" vector-effect="non-scaling-stroke"></line>
                  <line x1="0" y1="140" x2="400" y2="140" stroke="#dddcd3" stroke-width="1" vector-effect="non-scaling-stroke"></line>
                  ${(g.marcas || []).map((k) => html`
                    <line x1=${k.x} y1="0" x2=${k.x} y2="140" stroke="#ecebe5" stroke-width="1" vector-effect="non-scaling-stroke"></line>
                  `)}
                  ${(g.hayRef) ? html`
                    <line x1="0" y1=${g.refY} x2="400" y2=${g.refY} stroke="#a3a99d" stroke-width="1" vector-effect="non-scaling-stroke"></line>
                  ` : null}
                  <polyline points=${g.puntos} fill="none" stroke="#0f6f5c" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"></polyline>
                </svg>
                ${(g.hayRef) ? html`
                  <span aria-hidden="true" style=${g.refEtiquetaEstilo}>${g.refEtiqueta}</span>
                ` : null}
                <span aria-hidden="true" style=${g.cursorEstilo}></span>
                <span aria-hidden="true" style=${g.puntoEstilo}></span>
                <span aria-hidden="true" style=${g.tooltipEstilo}>${g.tooltip}</span>
                <div onMouseMove=${v.moverGrafica} onMouseLeave=${v.salirGrafica} style="position: absolute; inset: 0; cursor: crosshair"></div>
              </div>
              <span></span>
              <div aria-hidden="true" style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #767771; font-variant-numeric: tabular-nums">
                <span>0</span><span>${g.xMitad}</span><span>${g.xFin}</span>
              </div>
            </div>
          </figure>
        `)}
      </div>
    </section>
  </main>
</div>`;
}

