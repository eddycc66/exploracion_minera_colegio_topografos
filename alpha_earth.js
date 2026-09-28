////////////////////////////////////////////////////////////////////////////////
// SPATIALMIND S.R.L. — ALPHAEARTH · SIMILITUD SEMI-SUPERVISADA · v1.1 (GEE JS)
////////////////////////////////////////////////////////////////////////////////
// v1.1 — MÓDULO DE ESTADÍSTICAS/DIAGNÓSTICO:
//   · FIX: bandas A00–A63 del embedding ya NO se detectan como "leyes"
//   · FIX: zonas a revisar (faltaba geometries:true → salían siempre 0)
//   · Matriz coseno entre puntos + sim de cada punto a la firma
//   · LOO por punto (detecta la muestra más atípica)
//   · Histograma ASCII del mapa de similitud + percentiles + % ROI >0.95/0.98
//   · Fondo: 2000 pts aleatorios (media, std, percentil donde caen tus puntos)
//   · Buffer 150 m vs resto del ROI · Fondo extendido (anillo 5 km)
//   · Diagnóstico de ROI pequeño (discriminación estructuralmente ~0)
//   · Ranking por discriminación + export CSV de estadísticas
//
// Atribución obligatoria (CC-BY 4.0):
//   The AlphaEarth Foundations Satellite Embedding dataset is produced by
//   Google and Google DeepMind.
////////////////////////////////////////////////////////////////////////////////

// ════════════════════════ 0 · CONFIGURACIÓN (edita solo esto) ════════════════
var ROI_ASSET    = 'projects/eddycc66/assets/area_PONGO_QUINUMA';
var SAMPLE_ASSET = 'projects/eddycc66/assets/muestras';
var AE_YEAR      = 2024;

var MODE            = 'auto';      // 'auto' | 'similitud' | 'clustering'
var USE_DEMO_POINTS = false;

var GRADE_WEIGHT_MODE = 'continuo';
var GRADE_THRESHOLDS  = {};
var MIN_GRADE_POINTS  = 2;
var MIN_POINTS_WARN   = 3;
var DISCRIM_MIN       = 0.10;
var SIM_PERCENTILE_HI = 90;

var N_DOMAINS       = 8;
var CLUSTER_SAMPLES = 5000;

// Diagnóstico espacial
var ROI_SMALL_KM2   = 10;          // ROI con menos área que esto = "pequeño"
var POINTS_CLOSE_M  = 300;         // puntos que abarcan menos que esto = "muy juntos"
var BG_RANDOM_N     = 2000;        // puntos aleatorios para la distribución de fondo
var RING_METERS     = 5000;        // anillo de "fondo extendido" fuera del ROI
var BUFFER_METERS   = 150;         // entorno inmediato de los puntos

var MINERAL_FIELDS = {'W': 'W (%)', 'Sn': 'Sn (%)', 'Pb': 'Pb (%)', 'Ag': 'Ag (g/t)'};

var NODATA   = ['-', '--', '', 'nd', 'n/a', 's/d', 'na', 'nan'];
var RESERVED = ['N°','N','DPTO','PROV','MUN','MUESTRA','COMUNIDAD','ZONA','DESCRIPCION',
                'DESCRIPCIO','DIP','INCLI','COMPOSICION MINERALOGICA','COMPOSICIO',
                'X','Y','id','system:index','.geo'];

var AE_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL';
var AE_ATTRIB  = 'AlphaEarth Foundations Satellite Embedding — Google & Google DeepMind (CC-BY 4.0)';
var PALETTE    = ['00E5FF','FFD700','FF9800','FF3B00'];
var MINERAL_COLORS = ['FF3B00','39B54A','00BCD4','FFB300','AB47BC','8BC34A','E91E63','607D8B'];
var CAT_PALETTE = ['e6194B','3cb44b','ffe119','4363d8','f58231','911eb4','42d4f4',
                   'f032e6','bfef45','fabed4','469990','dcbeff'];
var EXPORT_FOLDER = 'SPATIALMIND_GEE';

var bandas = [];
for (var ib = 0; ib < 64; ib++) bandas.push('A' + (ib < 10 ? '0' : '') + ib);
var BANDA_SET = {}; bandas.forEach(function(b){ BANDA_SET[b] = true; });  // FIX v1.1

// ════════════════════════ 1 · HELPERS (cliente) ══════════════════════════════
function safeGet(obj, label, def) {
  try { return obj.getInfo(); }
  catch (e) { print('  ⚠ ' + label + ': ' + String(e).slice(0, 100)); return def; }
}

function parseGrade(raw) {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === 'number') return (isNaN(raw) ? null : raw);
  var s = String(raw).toLowerCase().trim();
  if (NODATA.indexOf(s) >= 0) return null;
  s = s.replace(/,/g, '.').replace(/^[<>]+/, '').trim();
  var v = parseFloat(s);
  return (isNaN(v) ? null : v);
}

function norm(v) { var s = 0; for (var i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s) || 1e-9; }
function normalizeVec(v) { var n = norm(v); return v.map(function(x){ return x / n; }); }
function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function avg(a) { return a.reduce(function(x, y){ return x + y; }, 0) / (a.length || 1); }

function weightedMean(Vs, Ws) {
  var d = Vs[0].length, sig = [], wsum = 0, i, j;
  for (j = 0; j < d; j++) sig.push(0);
  for (i = 0; i < Vs.length; i++) {
    var w = Math.max(Ws[i], 0);
    for (j = 0; j < d; j++) sig[j] += Vs[i][j] * w;
    wsum += w;
  }
  if (wsum <= 0) { for (i = 0; i < Vs.length; i++) for (j = 0; j < d; j++) sig[j] += Vs[i][j]; wsum = Vs.length; }
  return normalizeVec(sig.map(function(x){ return x / wsum; }));
}

function signatureNorm(Vs, Ws) {  // ||Σ w·v|| / Σw ANTES de normalizar ≈ coherencia geométrica
  var d = Vs[0].length, s = [], ws = 0, i, j;
  for (j = 0; j < d; j++) s.push(0);
  for (i = 0; i < Vs.length; i++) { var w = Math.max(Ws[i], 0); ws += w;
    for (j = 0; j < d; j++) s[j] += Vs[i][j] * w; }
  return ws > 0 ? norm(s) / ws : 0;
}

function looConsistency(Vs, Ws) { // leave-one-out PONDERADO → {mean, vals, minIdx}
  if (Vs.length < 2) return null;
  var vals = [];
  for (var i = 0; i < Vs.length; i++) {
    var Vo = [], Wo = [];
    for (var j = 0; j < Vs.length; j++) if (j !== i) { Vo.push(Vs[j]); Wo.push(Ws[j]); }
    vals.push(dot(Vs[i], weightedMean(Vo, Wo)));
  }
  var mean = avg(vals), minIdx = 0;
  for (var k = 1; k < vals.length; k++) if (vals[k] < vals[minIdx]) minIdx = k;
  return {mean: mean, vals: vals, minIdx: minIdx};
}

function pairwiseCos(Vs, ids) {   // matriz coseno entre todos los pares de puntos
  var n = Vs.length, M = [], lowPairs = 0, sum = 0, cnt = 0, minV = 2, minPair = '';
  for (var i = 0; i < n; i++) {
    M.push([]);
    for (var j = 0; j < n; j++) {
      var c = (i === j) ? 1 : dot(Vs[i], Vs[j]);
      M[i].push(c);
      if (j < i) { sum += c; cnt++;
        if (c < minV) { minV = c; minPair = ids[i] + ' ↔ ' + ids[j]; }
        if (c < 0.5) lowPairs++; }
    }
  }
  return {M: M, mean: cnt ? sum / cnt : null, min: minV, minPair: minPair, lowPairs: lowPairs};
}

function distM(a, b) {            // distancia aproximada (áreas pequeñas)
  var R = 6371000, rad = Math.PI / 180;
  var x = (b.lon - a.lon) * rad * Math.cos(((a.lat + b.lat) / 2) * rad) * R;
  var y = (b.lat - a.lat) * rad * R;
  return Math.sqrt(x * x + y * y);
}

function fmt(v) { return (v === null || v === undefined || isNaN(v)) ? 's/d' : String(Math.round(v * 1000) / 1000); }
function fmtS(v) { return (v === null || v === undefined || isNaN(v)) ? 's/d' : (v >= 0 ? '+' : '') + v.toFixed(3); }
function pad(s, n) { s = String(s); while (s.length < n) s += ' '; return s; }
function nivelTxt(n) { return n === 'alto' ? 'FIABLE' : (n === 'medio' ? 'CON CAUTELA' : 'NO FIABLE'); }

function verdictOf(cons, disc) {
  if (cons === null)
    return {txt: 'Sólo hay un punto: no se puede validar la coherencia de la firma. Interpreta con máxima cautela.', nivel: 'bajo'};
  if (disc !== null && disc < DISCRIM_MIN)
    return {txt: '⚠ EL MAPA NO DISCRIMINA AQUÍ: puntos coherentes (consistencia ' + fmt(cons) +
                 ') pero se parecen a CASI TODA el área (discriminación ' + fmt(disc) +
                 '). Ver las ESTADÍSTICAS: ¿ROI muy pequeño? ¿cobertura homogénea? ' +
                 'Recomendación: geoquímica de suelos/sedimentos o geofísica.', nivel: 'bajo'};
  if (cons < 0.15)
    return {txt: '⚠ FIRMA INCOHERENTE: tus puntos NO se parecen entre sí. No uses el mapa; separa por tipo de ambiente y corre por grupo.', nivel: 'bajo'};
  if (cons < 0.40)
    return {txt: 'Coherencia moderada y firma discriminante: el mapa sirve como guía. Prioriza sólo la similitud relativa más alta.', nivel: 'medio'};
  return {txt: '✓ Firma coherente y discriminante: las zonas de alta similitud relativa son candidatas razonables para extrapolar y muestrear.', nivel: 'alto'};
}

function printRing(label, ring) {
  var lons = ring.map(function(p){ return p[0]; });
  var lats = ring.map(function(p){ return p[1]; });
  print('  ' + label + ': lon [' + Math.min.apply(null, lons).toFixed(5) + ', ' +
        Math.max.apply(null, lons).toFixed(5) + '] · lat [' + Math.min.apply(null, lats).toFixed(5) +
        ', ' + Math.max.apply(null, lats).toFixed(5) + ']');
}

function addInfoPanel(title, rows) {
  var panel = ui.Panel({style: {position: 'bottom-left', width: '330px'}});
  panel.add(ui.Label({value: title, style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'}}));
  rows.forEach(function(r) {
    if (r.swatch !== undefined) {
      panel.add(ui.Panel({widgets: [
          ui.Label('', {backgroundColor: r.swatch, padding: '8px', margin: '0 8px 4px 0'}),
          ui.Label(r.text, {margin: '0 0 4px 0', fontSize: '12px'})],
        layout: ui.Panel.Layout.flow('horizontal')}));
    } else {
      panel.add(ui.Label(r.text, {margin: '0 0 4px 0', fontSize: '12px',
                                  fontWeight: r.bold ? 'bold' : 'normal'}));
    }
  });
  panel.add(ui.Label('⚠ Similitud alta ≠ mineral: es un proxy de extrapolación. Verifica en terreno.\n' + AE_ATTRIB,
                     {fontSize: '10px', color: '555', margin: '8px 0 0 0'}));
  Map.add(panel);
}

// ════════════════════════ 2 · MAIN ═══════════════════════════════════════════
(function main() {

  print('══════════ SPATIALMIND · AlphaEarth · Similitud semi-supervisada v1.1 ══════════');
  print('▶ ROI: ' + ROI_ASSET.split('/').pop() + ' · asset de puntos: ' + (SAMPLE_ASSET || '(ninguno)'));

  var roi = ee.FeatureCollection(ROI_ASSET).geometry();
  var roiRing = safeGet(roi.bounds(), 'ROI bounds', null);
  if (roiRing) printRing('ROI', roiRing.coordinates[0]);
  Map.setOptions('SATELLITE');
  Map.centerObject(roi, 13);
  Map.addLayer(ee.FeatureCollection(roi), {color: '39B54A'}, 'Área de estudio (ROI)');

  // ── ALPHAEARTH (con fallback de año) ───────────────────────────────────────
  var emb = null, yearUsed = null;
  var years = [AE_YEAR, AE_YEAR - 1, AE_YEAR - 2, 2024, 2023];
  for (var yi = 0; yi < years.length && !emb; yi++) {
    var y = years[yi];
    var col = ee.ImageCollection(AE_COLLECTION).filterDate(y + '-01-01', (y + 1) + '-01-01').filterBounds(roi);
    var n = safeGet(col.size(), 'AlphaEarth ' + y, 0);
    if (n > 0) {
      print('▶ AlphaEarth: ' + n + ' imagen(es) para ' + y + ' · 64D @ 10 m');
      emb = col.mosaic().select(bandas).clip(roi);
      yearUsed = y;
    }
  }
  if (!emb) { print('✗ AlphaEarth no disponible para el ROI/año.'); return; }

  // ── PUNTOS DE CAMPO ────────────────────────────────────────────────────────
  var fc = null, isDemo = false;
  if (MODE !== 'clustering' && SAMPLE_ASSET) {
    try {
      var f0 = ee.FeatureCollection(SAMPLE_ASSET).filterBounds(roi);
      var n0 = f0.size().getInfo();
      if (n0 > 0) { fc = f0; print('▶ Puntos de campo (asset): ' + n0); }
    } catch (e) { print('  ⚠ No se pudo cargar SAMPLE_ASSET (' + String(e).slice(0, 70) + ').'); }
  }
  if (!fc && MODE !== 'clustering' && USE_DEMO_POINTS) {
    fc = ee.FeatureCollection.randomPoints(roi, 8, 42);
    ['W (%)','Sn (%)','Pb (%)','Ag (g/t)'].forEach(function(c, ci){ fc = fc.randomColumn(c, ci); });
    fc = fc.map(function(f) {
      return f.set('W (%)',  ee.Number(f.get('W (%)')).multiply(0.6))
              .set('Sn (%)', ee.Number(f.get('Sn (%)')).multiply(0.3))
              .set('Pb (%)', ee.Number(f.get('Pb (%)')).multiply(1.5))
              .set('Ag (g/t)', ee.Number(f.get('Ag (g/t)')).multiply(40));
    });
    isDemo = true;
    print('⚠ PUNTOS DEMO (aleatorios): la mecánica es real, los resultados NO tienen valor de exploración.');
  }

  // ═══ MODO CLUSTERING ═══════════════════════════════════════════════════════
  function runClustering() {
    print('▶ MODO CLUSTERING (sin puntos) · ' + N_DOMAINS + ' dominios. NO es prospectividad.');
    var meanDict = safeGet(emb.reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 30,
                    bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'medias', {}) || {};
    var stdDict  = safeGet(emb.reduceRegion({reducer: ee.Reducer.stdDev(), geometry: roi, scale: 30,
                    bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'desviaciones', {}) || {};
    var means = ee.Image.constant(bandas.map(function(b){ return (meanDict[b] !== undefined) ? meanDict[b] : 0; })).rename(bandas);
    var stds  = ee.Image.constant(bandas.map(function(b){ var s = stdDict[b]; return (s !== undefined && s > 0) ? s : 1; })).rename(bandas);
    var embZ  = emb.subtract(means).divide(stds.max(1e-6));

    var training  = embZ.sample({region: roi, scale: 10, numPixels: CLUSTER_SAMPLES, seed: 7, tileScale: 8, dropNulls: true});
    var clusterer = ee.Clusterer.wekaKMeans(N_DOMAINS).train(training);
    var dominios  = embZ.cluster(clusterer).rename('domain').clip(roi);
    Map.addLayer(dominios, {min: 0, max: N_DOMAINS - 1, palette: CAT_PALETTE.slice(0, N_DOMAINS), opacity: 0.6},
                 'Dominios del terreno (' + N_DOMAINS + ')');

    var hist = safeGet(dominios.reduceRegion({reducer: ee.Reducer.frequencyHistogram(), geometry: roi, scale: 30,
              bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'tamaño de dominios', {}) || {};
    var h = hist['domain'] || {};
    var total = 0; Object.keys(h).forEach(function(k){ total += h[k]; });
    if (!total) total = 1;
    var raros = [];
    Object.keys(h).sort(function(a, b){ return (+a) - (+b); }).forEach(function(k) {
      var frac = h[k] / total, raro = frac < 0.5 / N_DOMAINS;
      if (raro) raros.push('D' + k);
      print('  Dominio ' + k + ': ' + Math.round(frac * 100) + '%' + (raro ? '  ⚠ raro' : ''));
    });
    if (raros.length) print('  ⚠ Dominios raros/atípicos (revisar primero): ' + raros.join(', '));
    exportImage(dominios.toInt16(), 'AlphaEarth_Dominios');
    print('─ Siguiente paso: muestrea 1+ punto por dominio (primero los raros) y vuelve a correr en modo similitud.');
    print('   ' + AE_ATTRIB);
  }

  function exportImage(img, desc) {
    Export.image.toDrive({image: img, description: desc, folder: EXPORT_FOLDER,
                          region: roi, scale: 10, crs: 'EPSG:4326', maxPixels: 1e13});
    print('  ⤓ Tarea: ' + desc + ' → pestaña Tasks → RUN');
  }
  function exportHotspots(hot, desc) {
    if (!hot || !hot.length) return;
    var fcs = ee.FeatureCollection(hot.map(function(h) {
      return ee.Feature(ee.Geometry.Point([h.lon, h.lat]), {id: h.id, sim: h.sim});
    }));
    Export.table.toDrive({collection: fcs, description: desc, folder: EXPORT_FOLDER, fileFormat: 'GeoJSON'});
    print('  ⤓ Tarea: ' + desc + ' (GeoJSON) → pestaña Tasks → RUN');
  }
  function exportCSV(rows, desc) {
    if (!rows.length) return;
    var fcs = ee.FeatureCollection(rows.map(function(s){ return ee.Feature(null, s); }));
    Export.table.toDrive({collection: fcs, description: desc, folder: EXPORT_FOLDER, fileFormat: 'CSV'});
    print('  ⤓ Tarea: ' + desc + '.csv (estadísticas por mineral) → pestaña Tasks → RUN');
  }

  // ═══ NÚCLEO DE SIMILITUD ═══════════════════════════════════════════════════
  function similarityMaps(sig) {
    var sigImg = ee.Image.constant(sig).rename(bandas);
    var sim = emb.multiply(sigImg).reduce(ee.Reducer.sum())
                .add(1).divide(2).clamp(0, 1).rename('AE_Similarity');
    var pr = safeGet(sim.reduceRegion({reducer: ee.Reducer.percentile([2, 98]), geometry: roi, scale: 60,
             bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'percentiles p2/p98', {}) || {};
    var p2  = (pr['AE_Similarity_p2']  !== undefined) ? pr['AE_Similarity_p2']  : 0;
    var p98 = (pr['AE_Similarity_p98'] !== undefined) ? pr['AE_Similarity_p98'] : 1;
    if (p98 <= p2) p98 = p2 + 1e-4;
    var rel = sim.unitScale(p2, p98).clamp(0, 1).rename('AE_Similarity_rel');
    return {abs: sim, rel: rel};
  }

  // FIX v1.1: geometries:true + umbral de respaldo (p90 → p75 → p60)
  function extractHotspots(simRel, prefix) {
    var thTry = [SIM_PERCENTILE_HI, 75, 60];
    for (var t = 0; t < thTry.length; t++) {
      var thDict = safeGet(simRel.reduceRegion({reducer: ee.Reducer.percentile([thTry[t]]), geometry: roi,
                   scale: 30, bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'umbral p' + thTry[t], {}) || {};
      var thi = thDict['AE_Similarity_rel_p' + thTry[t]];
      if (thi === undefined) continue;
      var hi = simRel.updateMask(simRel.gte(thi));
      var peaks = hi.gte(hi.focalMax(90, 'circle', 'meters')).selfMask();
      var hs = simRel.rename('sim').updateMask(peaks)
               .sample({region: roi, scale: 10, numPixels: 6000, seed: 7, dropNulls: true,
                        geometries: true, tileScale: 8});                       // ← FIX clave
      var raw = (safeGet(hs, 'picos p' + thTry[t], {features: []}).features) || [];
      if (!raw.length) continue;
      if (!roiRing) return [];
      var ring = roiRing.coordinates[0];
      var lonMn = Math.min.apply(null, ring.map(function(p){ return p[0]; }));
      var lonMx = Math.max.apply(null, ring.map(function(p){ return p[0]; }));
      var latMn = Math.min.apply(null, ring.map(function(p){ return p[1]; }));
      var latMx = Math.max.apply(null, ring.map(function(p){ return p[1]; }));
      var cells = {};
      raw.forEach(function(f) {
        var s = f.properties ? f.properties.sim : null;
        if (s === null || s === undefined || !f.geometry) return;
        var lon = f.geometry.coordinates[0], lat = f.geometry.coordinates[1];
        var r = Math.min(2, Math.floor((lat - latMn) / ((latMx - latMn) / 3 + 1e-12)));
        var c = Math.min(2, Math.floor((lon - lonMn) / ((lonMx - lonMn) / 3 + 1e-12)));
        var k = r + '_' + c;
        if (!cells[k]) cells[k] = [];
        cells[k].push([s, lon, lat]);
      });
      var HOT = [];
      Object.keys(cells).forEach(function(k) {
        cells[k].sort(function(a, b){ return b[0] - a[0]; });
        cells[k].slice(0, 4).forEach(function(p) {
          HOT.push({sim: Math.round(p[0] * 1000) / 1000, lon: p[1], lat: p[2]});
        });
      });
      HOT.sort(function(a, b){ return b.sim - a.sim; });
      HOT.forEach(function(h, i){ h.id = prefix + '-' + ('00' + (i + 1)).slice(-3); });
      if (t > 0) print('    (umbral relajado a p' + thTry[t] + ' para obtener zonas)');
      return HOT;
    }
    return [];
  }

  // ── ESTADÍSTICAS ───────────────────────────────────────────────────────────
  function simPercentiles(simImg) {
    return safeGet(simImg.reduceRegion({
      reducer: ee.Reducer.percentile([1, 5, 25, 50, 75, 95, 99]),
      geometry: roi, scale: 30, bestEffort: true, maxPixels: 1e10, tileScale: 8}),
      'percentiles mapa', {}) || {};
  }
  function asciiHistogram(simImg, label) {
    var d = safeGet(simImg.reduceRegion({reducer: ee.Reducer.fixedHistogram(0, 1, 20), geometry: roi,
           scale: 30, bestEffort: true, maxPixels: 1e10, tileScale: 8}), label, null);
    var arr = d ? d['AE_Similarity'] : null;
    if (!arr) return;
    var maxC = 0, total = 0;
    arr.forEach(function(p){ total += p[1]; if (p[1] > maxC) maxC = p[1]; });
    print('  Histograma de similitud en el ROI (20 bins · escala absoluta 0–1):');
    for (var i = 0; i < arr.length; i++) {
      var lo = arr[i][0], c = arr[i][1];
      if (!c) continue;
      var nBars = Math.max(1, Math.round(38 * c / (maxC || 1)));
      var bar = ''; for (var j = 0; j < nBars; j++) bar += '█';
      print('    ' + lo.toFixed(2) + '–' + (lo + 0.05).toFixed(2) + ' │' + bar + ' ' +
            (100 * c / (total || 1)).toFixed(1) + '%');
    }
  }
  function fracAbove(simImg, th) {
    var s = safeGet(simImg.gt(th).reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 60,
           bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'frac >' + th, {}) || {};
    return (s['AE_Similarity'] !== undefined) ? s['AE_Similarity'] : null;
  }
  function backgroundStats(simImg, meanPts) {
    var pts = ee.FeatureCollection.randomPoints(roi, BG_RANDOM_N, 123);
    var rr = safeGet(simImg.reduceRegions({collection: pts, reducer: ee.Reducer.mean(), scale: 10, tileScale: 8}),
                     'fondo aleatorio', null);
    if (!rr || !rr.features) return null;
    var vals = rr.features.map(function(f){ return f.properties ? f.properties.mean : null; })
                          .filter(function(x){ return x !== null && x !== undefined; });
    if (!vals.length) return null;
    var mean = avg(vals);
    var std = Math.sqrt(avg(vals.map(function(v){ return (v - mean) * (v - mean); })));
    var sorted = vals.slice().sort(function(a, b){ return a - b; });
    function pctile(q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]; }
    var below = sorted.filter(function(v){ return v < meanPts; }).length / sorted.length;
    return {n: vals.length, mean: mean, std: std, p50: pctile(0.5), p90: pctile(0.9),
            max: sorted[sorted.length - 1], pctBelowMeanPts: below};
  }
  function bufferVsRest(simImg) {
    try {
      var bufGeom = fc.map(function(f){ return f.buffer(BUFFER_METERS); }).geometry(1);
      var mBuf = safeGet(simImg.reduceRegion({reducer: ee.Reducer.mean(), geometry: bufGeom, scale: 10,
                 bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'buffer', {}) || {};
      var restGeom = roi.difference(bufGeom, 1);
      var mRest = safeGet(simImg.reduceRegion({reducer: ee.Reducer.mean(), geometry: restGeom, scale: 30,
                  bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'resto', {}) || {};
      return {buf: mBuf['AE_Similarity'], rest: mRest['AE_Similarity']};
    } catch (e) { return null; }
  }
  function extendedBackground(simImg) {
    try {
      var ringGeom = roi.bounds().buffer(RING_METERS);
      ringGeom = ringGeom.difference(roi, 1);
      var m = safeGet(simImg.reduceRegion({reducer: ee.Reducer.mean(), geometry: ringGeom, scale: 60,
              bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'fondo extendido', {}) || {};
      return (m['AE_Similarity'] !== undefined) ? m['AE_Similarity'] : null;
    } catch (e) { return null; }
  }

  // Validación + TODAS las estadísticas de un mineral
  function validateAndReport(key, idxs, Vs, Ws, sig, maps, prefix) {
    var loo = looConsistency(Vs, Ws);
    var cons = loo ? loo.mean : null;
    var subFc = ee.FeatureCollection(idxs.map(function(i) {
      return ee.Feature(ee.Geometry.Point([P[i].lon, P[i].lat])); }));
    var meanPts = null;
    var rr = safeGet(maps.abs.reduceRegions({collection: subFc, reducer: ee.Reducer.mean(), scale: 10, tileScale: 8}),
                     'similitud en puntos', null);
    if (rr && rr.features) {
      var vals = rr.features.map(function(f){ return f.properties ? f.properties.mean : null; })
                            .filter(function(x){ return x !== null && x !== undefined; });
      if (vals.length) meanPts = avg(vals);
    }
    var bg = safeGet(maps.abs.reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 60,
            bestEffort: true, maxPixels: 1e10, tileScale: 8}), 'similitud de fondo', {}) || {};
    var meanBg = (bg['AE_Similarity'] !== undefined) ? bg['AE_Similarity'] : null;
    var disc = (meanPts !== null && meanBg !== null) ? (meanPts - meanBg) : null;
    var v = verdictOf(cons, disc);
    print('    Consistencia (LOO): ' + fmt(cons) + ' · Discriminación: ' + fmtS(disc) +
          '  (puntos ' + fmt(meanPts) + ' − fondo ROI ' + fmt(meanBg) + ')');
    print('    → ' + v.txt);

    // ══ BLOQUE DE ESTADÍSTICAS ══
    print('    ── ESTADÍSTICAS · ' + key + ' ──');
    var ids = idxs.map(function(i){ return idOf(i); });

    // (1) pesos usados
    var wtxt = idxs.map(function(i, k){ return ids[k] + '=' + Number(Ws[k]).toFixed(3); }).join(', ');
    print('    Peso por muestra (' + GRADE_WEIGHT_MODE + '): ' + wtxt);

    // (2) matriz coseno entre puntos
    if (Vs.length >= 2 && Vs.length <= 8) {
      var pw = pairwiseCos(Vs, ids);
      print('    Matriz coseno entre puntos (media ' + fmt(pw.mean) + ' · mín ' + fmt(pw.min) +
            ' en ' + pw.minPair + (pw.lowPairs ? ' · ⚠ ' + pw.lowPairs + ' par(es) < 0.5' : '') + '):');
      var head = '        '; ids.forEach(function(id){ head += pad(id.slice(0, 7), 8); });
      print(head);
      for (var i = 0; i < Vs.length; i++) {
        var row = '    ' + pad(ids[i].slice(0, 4), 6) + ' ';
        for (var j = 0; j < Vs.length; j++) row += pad(pw.M[i][j].toFixed(3), 8);
        print(row);
      }
    }

    // (3) norma de la firma antes de normalizar (≈ alineación de los vectores)
    print('    Norma de la firma antes de normalizar: ' + fmt(signatureNorm(Vs, Ws)) +
          '  (1 = vectores idénticos · bajo = dispersos)');

    // (4) similitud de cada punto a la firma final
    var simsSig = Vs.map(function(v){ return dot(v, sig); });
    var pairs = []; for (var s = 0; s < simsSig.length; s++) pairs.push(ids[s] + ': ' + simsSig[s].toFixed(3));
    print('    Sim. de cada punto a la FIRMA: ' + pairs.join(' · '));

    // (5) LOO por punto → la muestra más atípica
    if (loo) {
      var lp = []; for (var q = 0; q < loo.vals.length; q++) lp.push(ids[q] + ': ' + loo.vals[q].toFixed(3));
      print('    LOO por punto: ' + lp.join(' · '));
      print('    ⤷ Muestra más atípica: ' + ids[loo.minIdx] + ' (LOO ' + loo.vals[loo.minIdx].toFixed(3) + ')');
    }

    // (6) percentiles + histograma del mapa (ABSOLUTO)
    var pc = simPercentiles(maps.abs);
    print('    Mapa ABSOLUTO · p1=' + fmt(pc['AE_Similarity_p1']) + ' · p25=' + fmt(pc['AE_Similarity_p25']) +
          ' · p50=' + fmt(pc['AE_Similarity_p50']) + ' · p75=' + fmt(pc['AE_Similarity_p75']) +
          ' · p95=' + fmt(pc['AE_Similarity_p95']) + ' · p99=' + fmt(pc['AE_Similarity_p99']));
    asciiHistogram(maps.abs, 'hist ' + key);

    // (7) fracción del ROI por encima de umbrales altos
    var f95 = fracAbove(maps.abs, 0.95), f98 = fracAbove(maps.abs, 0.98);
    print('    Fracción del ROI con sim > 0.95: ' + (f95 === null ? 's/d' : (f95 * 100).toFixed(1) + '%') +
          ' · > 0.98: ' + (f98 === null ? 's/d' : (f98 * 100).toFixed(1) + '%') +
          (f95 !== null && f95 > 0.6 ? '  ⚠ casi todo el ROI es muy similar → no discrimina' : ''));

    // (8) fondo aleatorio (2000 pts) + percentil donde caen tus puntos
    var bgS = backgroundStats(maps.abs, meanPts);
    if (bgS) {
      print('    Fondo aleatorio (' + bgS.n + ' pts): media ' + bgS.mean.toFixed(3) + ' ± ' + bgS.std.toFixed(3) +
            ' · p50 ' + bgS.p50.toFixed(3) + ' · p90 ' + bgS.p90.toFixed(3) + ' · máx ' + bgS.max.toFixed(3));
      print('    Tus puntos caen por encima del ' + Math.round(bgS.pctBelowMeanPts * 100) +
            '% del fondo (' + (bgS.pctBelowMeanPts > 0.7 ? 'separación aceptable' : 'prácticamente indistinguibles del fondo') + ')');
    }

    // (9) buffer 150 m vs resto del ROI (estructura espacial)
    var br = bufferVsRest(maps.abs);
    if (br) print('    Entorno inmediato (buffer ' + BUFFER_METERS + ' m): ' + fmt(br.buf) +
                  ' · resto del ROI: ' + fmt(br.rest) + ' · Δ ' + fmtS((br.buf !== null && br.rest !== null) ? br.buf - br.rest : null));

    // (10) fondo extendido (anillo fuera del ROI) — clave en ROIs pequeños
    var bgExt = extendedBackground(maps.abs);
    if (bgExt !== null && meanPts !== null)
      print('    Fondo EXTENDIDO (anillo ' + (RING_METERS / 1000) + ' km fuera del ROI): ' + bgExt.toFixed(3) +
            ' → discriminación vs este fondo: ' + fmtS(meanPts - bgExt) +
            ((meanPts - bgExt) >= DISCRIM_MIN ? '  ✓ aquí SÍ discrimina' : ''));

    var hot = extractHotspots(maps.rel, prefix);
    print('    ✓ Zonas de alta similitud: ' + hot.length);
    return {cons: cons, disc: disc, meanPts: meanPts, meanBg: meanBg, bgExt: bgExt,
            buffer: br ? br.buf : null, rest: br ? br.rest : null,
            p50: pc['AE_Similarity_p50'], p95: pc['AE_Similarity_p95'],
            f95: f95, loo: loo, bgS: bgS, nivel: v.nivel, veredicto: v.txt, hot: hot};
  }

  // ═══ MODO SIMILITUD ════════════════════════════════════════════════════════
  if (MODE === 'clustering' || (MODE === 'auto' && !fc)) { runClustering(); return; }
  if (!fc) { print('✗ MODE="similitud" pero no hay puntos: define SAMPLE_ASSET o activa USE_DEMO_POINTS.'); return; }

  Map.addLayer(fc.style({color: '#FFEB3B', fillColor: '#FFEB3B', pointSize: 7}), {},
               'Puntos de campo' + (isDemo ? ' (DEMO)' : ''));
  var pRing = safeGet(fc.geometry().bounds(), 'bounds puntos', null);
  if (pRing) printRing('Puntos de campo', pRing.coordinates[0]);

  // Extraer embeddings 64D en los puntos
  var ptsEmb = emb.sampleRegions({collection: fc, scale: 10, geometries: true, tileScale: 8});
  var feats = (safeGet(ptsEmb, 'embeddings en puntos', {features: []}).features) || [];
  var P = [];
  feats.forEach(function(f) {
    var v = [], ok = true;
    for (var d = 0; d < 64; d++) {
      var x = f.properties ? f.properties[bandas[d]] : null;
      if (x === null || x === undefined) { ok = false; break; }
      v.push(x);
    }
    if (ok && f.geometry && f.geometry.coordinates)
      P.push({v: normalizeVec(v), lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1], props: f.properties});
  });
  print('▶ Puntos sobre datos de AlphaEarth: ' + P.length + ' de ' + feats.length);
  if (P.length === 0) {
    print('✗ Ningún punto cayó sobre AlphaEarth. Compara los rangos impresos arriba (ROI vs Puntos).');
    print('  Si no se solapan: revisa el CRS del asset (UTM subido como grados es el error más común).');
    return;
  }
  print('▶ MODO SIMILITUD · Puntos de muestreo (firma): ' + P.length);
  if (P.length < MIN_POINTS_WARN)
    print('  ⚠ Sólo ' + P.length + ' punto(s): la firma será poco fiable.');

  // ── DIAGNÓSTICO ESPACIAL GLOBAL (nuevo v1.1) ───────────────────────────────
  var roiAreaKm2 = null;
  var aInfo = safeGet(roi.area(1), 'área ROI', null);
  if (aInfo) roiAreaKm2 = aInfo / 1e6;
  var lons = P.map(function(p){ return p.lon; }), lats = P.map(function(p){ return p.lat; });
  var lonSpanKm = (Math.max.apply(null, lons) - Math.min.apply(null, lons)) * 111.32 *
                  Math.cos(avg(lats) * Math.PI / 180);
  var latSpanKm = (Math.max.apply(null, lats) - Math.min.apply(null, lats)) * 111.32;
  var maxPair = 0;
  for (var di = 0; di < P.length; di++)
    for (var dj = di + 1; dj < P.length; dj++) {
      var dd = distM(P[di], P[dj]); if (dd > maxPair) maxPair = dd;
    }
  var roiSmall = (roiAreaKm2 !== null && roiAreaKm2 < ROI_SMALL_KM2);
  var ptsClose = (maxPair * 1000 < POINTS_CLOSE_M);
  print('── DIAGNÓSTICO ESPACIAL ──');
  print('  ROI: ' + (roiAreaKm2 ? roiAreaKm2.toFixed(2) + ' km²' : 's/d') +
        ' · puntos: dispersión máx ' + (maxPair).toFixed(0) + ' m · bbox ' +
        (lonSpanKm * 1000).toFixed(0) + ' × ' + (latSpanKm * 1000).toFixed(0) + ' m');
  if (roiSmall)
    print('  ⚠ ROI PEQUEÑO (< ' + ROI_SMALL_KM2 + ' km²): el "fondo" del ROI es prácticamente el entorno' +
          ' inmediato de tus puntos → la discriminación tenderá a ~0 AUNQUE la firma sea buena.' +
          ' Para una prueba justa, corre con un ROI distrital (decenas de km²) y/o mira la métrica' +
          ' "fondo EXTENDIDO (anillo ' + (RING_METERS / 1000) + ' km)" en las estadísticas de cada mineral.');
  if (ptsClose)
    print('  ⚠ Puntos muy juntos (< ' + POINTS_CLOSE_M + ' m): la firma describe UN solo ambiente local,' +
          ' no un depósito. Si tienes muestras de afloramientos distintos, inclúyelas para enriquecer la firma.');

  // ── Leyes (FIX v1.1: A00–A63 y system:index NUNCA son leyes) ───────────────
  function detectMineralFields(propsList) {
    var fields = {}, mapped = {}, allCols = {};
    propsList.forEach(function(p){ Object.keys(p || {}).forEach(function(c){ allCols[c] = true; }); });
    Object.keys(MINERAL_FIELDS).forEach(function(k) {
      var col = MINERAL_FIELDS[k];
      if (allCols[col]) { fields[k] = col; mapped[col] = true; }
    });
    Object.keys(allCols).forEach(function(c) {
      if (mapped[c] || RESERVED.indexOf(c) >= 0 || BANDA_SET[c] || /^A\d\d$/.test(c)) return;
      var n = 0;
      propsList.forEach(function(p){ if (parseGrade((p || {})[c]) !== null) n++; });
      if (n >= MIN_GRADE_POINTS) {
        var key = c.replace(/\s*\(.*?\)\s*/g, '').trim() || c;
        if (!fields[key]) { fields[key] = c; mapped[c] = true; }
      }
    });
    return fields;
  }

  function idOf(i) {
    var pr = P[i].props || {};
    var v = pr['MUESTRA'] || pr['muestra'] || pr['ID'] || pr['id'];
    if (v === undefined || v === null || v === '') v = pr['system:index'] || ('P' + (i + 1));
    return String(v);
  }

  var propsList = P.map(function(p){ return p.props; });
  var mineralFields = detectMineralFields(propsList);
  var keys = Object.keys(mineralFields);
  var grades = P.map(function(p) {
    var g = {}; keys.forEach(function(k){ g[k] = parseGrade(p.props[mineralFields[k]]); }); return g;
  });
  print('▶ Leyes detectadas: ' + (keys.length ? keys.join(', ') : '(ninguna)') + (isDemo ? '  (DEMO)' : ''));

  function addHotLayer(hot, color, layerName) {
    if (!hot || !hot.length) return;
    var hotFc = ee.FeatureCollection(hot.map(function(h) {
      return ee.Feature(ee.Geometry.Point([h.lon, h.lat]), {id: h.id, sim: h.sim});
    }));
    Map.addLayer(hotFc.style({color: '#' + color, fillColor: '#' + color, pointSize: 9}), {}, layerName);
  }

  var statsRows = [];   // para el CSV de estadísticas

  function runMineral(key, colName, ki) {
    var thr = (GRADE_WEIGHT_MODE === 'binario') ? (GRADE_THRESHOLDS[key] || 0) : 0;
    var idxs = [];
    for (var i = 0; i < P.length; i++) {
      var g = grades[i][key];
      if (g !== null && g > thr) idxs.push(i);
    }
    if (idxs.length < MIN_GRADE_POINTS) {
      print('  ⚠ ' + key + ' (' + colName + '): omitido — sólo ' + idxs.length +
            ' muestra(s) con ley > 0 (mínimo ' + MIN_GRADE_POINTS + ').');
      return null;
    }
    print('▶ MODO SIMILITUD · ' + key + ' (' + colName + ') · ' + idxs.length + ' muestra(s) con ley > 0' +
          (idxs.length < MIN_POINTS_WARN ? '  ⚠ pocas muestras' : ''));
    var Vs = idxs.map(function(i){ return P[i].v; });
    var Ws = idxs.map(function(i){ return (GRADE_WEIGHT_MODE === 'continuo') ? grades[i][key] : 1.0; });
    var sig = weightedMean(Vs, Ws);
    var maps = similarityMaps(sig);
    var val = validateAndReport(key, idxs, Vs, Ws, sig, maps, 'SM-AE-' + key);
    var color = MINERAL_COLORS[ki % MINERAL_COLORS.length];
    Map.addLayer(maps.rel, {min: 0, max: 1, palette: PALETTE, opacity: 0.6}, 'Similitud — ' + key, false);
    addHotLayer(val.hot, color, 'Zonas a revisar — ' + key);
    exportHotspots(val.hot, 'AlphaEarth_Zonas_' + key);
    exportImage(maps.abs, 'AlphaEarth_Similitud_' + key);
    var st = {mineral: key, columna: colName, n_muestras: idxs.length,
              consistencia: val.cons, discriminacion: val.disc, sim_media_puntos: val.meanPts,
              fondo_ROI: val.meanBg, fondo_ext_ring5km: val.bgExt, buffer_150m: val.buffer,
              resto_ROI: val.rest, mapa_p50: val.p50, mapa_p95: val.p95,
              pct_ROI_gt_095: val.f95, nivel: val.nivel, zonas: val.hot.length};
    Object.keys(st).forEach(function(k){ if (typeof st[k] === 'number') st[k] = Math.round(st[k] * 1e4) / 1e4; });
    statsRows.push(st);
    return {key: key, col: colName, n: idxs.length, cons: val.cons, disc: val.disc,
            nivel: val.nivel, veredicto: val.veredicto, color: color, rel: maps.rel, stats: st};
  }

  var resultados = [];
  keys.forEach(function(k, ki) {
    var r = runMineral(k, mineralFields[k], ki);
    if (r) resultados.push(r);
  });

  if (!resultados.length) {
    print('✗ Ningún mineral con suficientes muestras (mín ' + MIN_GRADE_POINTS + ') → firma única.');
    var Vs1 = P.map(function(p){ return p.v; }), Ws1 = Vs1.map(function(){ return 1.0; });
    var idx1 = Vs1.map(function(_, i){ return i; });
    var sig1 = weightedMean(Vs1, Ws1);
    var maps1 = similarityMaps(sig1);
    var val1 = validateAndReport('(firma única)', idx1, Vs1, Ws1, sig1, maps1, 'SM-AE');
    Map.addLayer(maps1.rel, {min: 0, max: 1, palette: PALETTE, opacity: 0.6}, 'Similitud (contraste relativo)');
    addHotLayer(val1.hot, 'FF3B00', 'Zonas a revisar');
    exportHotspots(val1.hot, 'AlphaEarth_Zonas');
    exportImage(maps1.abs, 'AlphaEarth_Similitud');
    addInfoPanel('SPATIALMIND · AlphaEarth — Similitud', [
      {text: 'Puntos: ' + P.length + ' · AlphaEarth ' + yearUsed, bold: true},
      {text: 'Consistencia: ' + fmt(val1.cons) + ' · Discriminación: ' + fmtS(val1.disc) + ' — ' + nivelTxt(val1.nivel)},
      {text: val1.veredicto},
      {swatch: '#00E5FF', text: 'Baja'}, {swatch: '#FFD700', text: 'Media'},
      {swatch: '#FF9800', text: 'Alta'}, {swatch: '#FF3B00', text: 'Muy alta → revisar'}
    ]);
    statsRows.push({mineral: '(firma única)', n_muestras: P.length, consistencia: val1.cons,
                    discriminacion: val1.disc, nivel: val1.nivel, zonas: val1.hot.length});
  } else {
    if (resultados.length > 1) {
      var combo = ee.ImageCollection.fromImages(resultados.map(function(r){ return r.rel; })).max();
      Map.addLayer(combo, {min: 0, max: 1, palette: PALETTE, opacity: 0.6},
                   'Índice combinado (máx entre minerales)');
    }
    // ── RANKING POR DISCRIMINACIÓN + LECTURA GLOBAL ──
    var rank = resultados.slice().sort(function(a, b){ return ((b.disc === null ? -9 : b.disc) - (a.disc === null ? -9 : a.disc)); });
    print('──────────────────────────────────────────────');
    print('▶▶ RANKING POR DISCRIMINACIÓN (mejor firma primero):');
    rank.forEach(function(r, i) {
      print('  ' + (i + 1) + '. ' + pad(r.key, 6) + ' n=' + r.n + ' · disc=' + fmtS(r.disc) +
            ' · consist=' + fmt(r.cons) + ' · ' + nivelTxt(r.nivel) +
            (r.stats && r.stats.fondo_ext_ring5km !== null && r.stats.fondo_ext_ring5km !== undefined && r.disc !== null &&
             (r.stats.meanPts - r.stats.fondo_ext_ring5km) >= DISCRIM_MIN ? '  ✓ discrimina vs fondo extendido' : ''));
    });
    var maxDisc = rank.length ? rank[0].disc : null;
    if (maxDisc !== null && maxDisc < DISCRIM_MIN) {
      print('▶▶ LECTURA GLOBAL: NINGÚN mineral discrimina (máx disc = ' + fmtS(maxDisc) + '). ' +
            (roiSmall ? 'Tu ROI (' + (roiAreaKm2 ? roiAreaKm2.toFixed(1) : '?') + ' km²) es pequeño y homogéneo: ' +
             'el fondo ≈ el entorno de los puntos. PRIMERO amplía el ROI a escala distrital y vuelve a correr — ' +
             'es la prueba más barata antes de descartar el método. Si con ROI grande tampoco discrimina, ' +
             'entonces es cobertura homogénea (AlphaEarth ve dosel, no roca) → geoquímica/geofísica.'
             : 'Cobertura homogénea: AlphaEarth ve el dosel, no la roca → geoquímica de suelos/sedimentos o geofísica.'));
    }
    print('   ' + AE_ATTRIB);
    print('──────────────────────────────────────────────');

    var rows = [{text: 'VALIDACIÓN POR MINERAL', bold: true}];
    rank.forEach(function(r) {
      rows.push({swatch: '#' + r.color,
                 text: r.key + ' · ' + r.n + ' muestras · consist ' + fmt(r.cons) +
                       ' · disc ' + fmtS(r.disc) + ' — ' + nivelTxt(r.nivel)});
    });
    rows.push({text: 'SIMILITUD (contraste relativo)', bold: true});
    rows.push({swatch: '#00E5FF', text: 'Baja'});
    rows.push({swatch: '#FFD700', text: 'Media'});
    rows.push({swatch: '#FF9800', text: 'Alta'});
    rows.push({swatch: '#FF3B00', text: 'Muy alta → zonas a revisar'});
    addInfoPanel('SPATIALMIND · AlphaEarth — Similitud por mineral', rows);
  }

  exportCSV(statsRows, 'AlphaEarth_Estadisticas');
})();