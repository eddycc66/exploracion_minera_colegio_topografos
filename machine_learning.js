// =========================================================================
// CLASIFICACIÓN MINERALÓGICA EN CERRO RICO DE POTOSÍ
// Con ranking de importancia por clase + panel de control
// Basado en: Calle Condori, E. (2026). Caso de estudio — Cerro Rico, Potosí.
// =========================================================================

// -------------------------------------------------------------------------
// 1. DEFINICIÓN DEL ÁREA DE ESTUDIO Y MUESTRAS
// -------------------------------------------------------------------------
var areaEstudio = ee.FeatureCollection('projects/eddycc66/assets/area_PONGO_QUINUMA');
var dem = ee.Image('projects/eddycc66/assets/dem_wolfram_pongo_quinuma');
var muestrasLab = ee.FeatureCollection('projects/eddycc66/assets/muestra_laboratorio_pongo_quinuma');

Map.centerObject(areaEstudio, 13);

// -------------------------------------------------------------------------
// 2. ASIGNAR CLASES A LAS MUESTRAS (SOLO PARA ESTE EJEMPLO)
// -------------------------------------------------------------------------
var listaMuestras = muestrasLab.toList(muestrasLab.size());
var nMuestras = muestrasLab.size();

var muestrasEtiquetadas = ee.FeatureCollection(
  ee.List.sequence(0, nMuestras.subtract(1)).map(function(i) {
    var feature = ee.Feature(listaMuestras.get(i));
    return feature.set('clase_num', i);
  })
);

// -------------------------------------------------------------------------
// 3. CARGA Y PREPROCESAMIENTO DE IMÁGENES SENTINEL-2
// -------------------------------------------------------------------------
var fechas = ['2023-03-15', '2023-06-15', '2023-09-15', '2023-12-15'];

var enmascararNubes = function(image) {
  var scl = image.select('SCL');
  var mascara = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(1));
  return image.select(['B2','B3','B4','B8','B11','B12'])
              .updateMask(mascara)
              .divide(10000);
};

var crearCompuesto = function(fechaInicio, fechaFin) {
  var coleccion = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(areaEstudio)
    .filterDate(fechaInicio, fechaFin)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(enmascararNubes);
  
  return ee.Image(ee.Algorithms.If(
    coleccion.size().gt(0),
    coleccion.median(),
    ee.Image.constant([0,0,0,0,0,0]).rename(['B2','B3','B4','B8','B11','B12']).selfMask()
  ));
};

var compuestos = fechas.map(function(fecha) {
  var inicio = ee.Date(fecha);
  var fin = inicio.advance(1, 'month');
  return crearCompuesto(inicio, fin).set('fecha', fecha);
});

var imagenMultitemporal = ee.ImageCollection(compuestos).toBands();
var nombresBandas = [];
var nombresFechas = ['mar', 'jun', 'sep', 'dic'];
['B2','B3','B4','B8','B11','B12'].forEach(function(banda) {
  nombresFechas.forEach(function(fecha) {
    nombresBandas.push(banda + '_' + fecha);
  });
});
imagenMultitemporal = imagenMultitemporal.rename(nombresBandas);

// -------------------------------------------------------------------------
// 4. ÍNDICES ESPECTRALES
// -------------------------------------------------------------------------
var calcularIndices = function(imagen) {
  var indices = [];
  nombresFechas.forEach(function(fecha) {
    var b2 = imagen.select('B2_' + fecha);
    var b4 = imagen.select('B4_' + fecha);
    var b8 = imagen.select('B8_' + fecha);
    var b11 = imagen.select('B11_' + fecha);
    var b12 = imagen.select('B12_' + fecha);
    
    var cai = b11.divide(b12).rename('CAI_' + fecha);
    var pai = b8.divide(b11).rename('PAI_' + fecha);
    var ioi = b4.divide(b2).rename('IOI_' + fecha);
    var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI_' + fecha);
    
    indices.push(cai, pai, ioi, ndvi);
  });
  return imagen.addBands(ee.Image.cat(indices));
};

var imagenConIndices = calcularIndices(imagenMultitemporal);

// -------------------------------------------------------------------------
// 5. MUESTREO Y ENTRENAMIENTO
// -------------------------------------------------------------------------
var predictoras = imagenConIndices.bandNames();

var muestras = imagenConIndices.sampleRegions({
  collection: muestrasEtiquetadas,
  properties: ['clase_num'],
  scale: 10,
  geometries: false,
  tileScale: 4
});

var entrenamiento = muestras;

// -------------------------------------------------------------------------
// 6. CLASIFICADORES
// -------------------------------------------------------------------------
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.632,
  maxNodes: null,
  seed: 42
}).train({
  features: entrenamiento,
  classProperty: 'clase_num',
  inputProperties: predictoras
});
var clasificacionRF = imagenConIndices.classify(rf).clip(areaEstudio);

var svm = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10,
  decisionProcedure: 'Voting'
}).train({
  features: entrenamiento,
  classProperty: 'clase_num',
  inputProperties: predictoras
});
var clasificacionSVM = imagenConIndices.classify(svm).clip(areaEstudio);

var kmeans = ee.Clusterer.wekaKMeans({
  nClusters: 4,
  maxIterations: 50,
  seed: 42
}).train({
  features: entrenamiento,
  inputProperties: predictoras
});
var clasificacionKMeans = imagenConIndices.cluster(kmeans).clip(areaEstudio);

var demClip = dem.clip(areaEstudio);

// -------------------------------------------------------------------------
// 7. 🔑 JERARQUÍA DE IMPORTANCIA POR CLASE
// -------------------------------------------------------------------------
// Basado en el contexto geológico de Cerro Rico:
// A MAYOR PRIORIDAD → MAYOR INTERÉS EXPLORATORIO (target de mineralización)
// A MENOR PRIORIDAD → MENOR INTERÉS (o incluso distractores)

var infoClases = [
  {
    id: 0,
    nombre: 'Cuarzo-sericita',
    color: '#FFD700',
    prioridad: 1,                    // 1 = MÁS IMPORTANTE
    nivel: '★★★★★ CRÍTICA',
    significado: 'Zona fílica en cúpula. Asociada directamente a la ' +
                 'mineralización Ag-Sn. MÁXIMO interés exploratorio.',
    uso: 'Objetivo principal de exploración.'
  },
  {
    id: 1,
    nombre: 'Jarita',
    color: '#A0522D',
    prioridad: 2,
    nivel: '★★★★☆ ALTA',
    significado: 'Zona de oxidación con óxidos de Fe. Indica presencia ' +
                 'de sulfuros alterados cerca de la superficie.',
    uso: 'Guía indirecta hacia cuerpos mineralizados.'
  },
  {
    id: 2,
    nombre: 'Caolinita',
    color: '#8B4513',
    prioridad: 3,
    nivel: '★★★☆☆ MEDIA',
    significado: 'Alteración argílica avanzada en flancos. Marca los ' +
                 'bordes del sistema hidrotermal, pero menos mineralizada.',
    uso: 'Contexto del sistema, delimita bordes.'
  },
  {
    id: 3,
    nombre: 'Óxidos de Fe',
    color: '#FF4500',
    prioridad: 4,
    nivel: '★★☆☆☆ BAJA',
    significado: 'Costras de oxidación superficiales. Pueden estar ' +
                 'desacopladas de la mineralización en profundidad.',
    uso: 'Referencia, pero no prioritario.'
  },
  {
    id: 4,
    nombre: 'Roca fresca',
    color: '#808080',
    prioridad: 5,
    nivel: '★☆☆☆☆ MUY BAJA',
    significado: 'Roca sin alteración hidrotermal aparente. NO tiene ' +
                 'interés mineralógico en este contexto.',
    uso: 'Clase de control (background).'
  },
  {
    id: 5,
    nombre: 'Vegetación',
    color: '#228B22',
    prioridad: 6,
    nivel: '☆☆☆☆☆ NULA',
    significado: 'Cobertura vegetal. Enmascara la señal espectral del ' +
                 'sustrato geológico. NO es de interés.',
    uso: 'Distractor a descartar en el análisis.'
  }
];

// Extraer arrays de trabajo
var paleta = infoClases.map(function(c) { return c.color; });
var nombresClases = infoClases.map(function(c) { return c.nombre; });

// -------------------------------------------------------------------------
// 8. VISUALIZACIÓN DE CAPAS
// -------------------------------------------------------------------------
var visClas = {min: 0, max: 5, palette: paleta};
var visDEM = {min: 3800, max: 4800, palette: ['blue', 'green', 'yellow', 'red']};

var capaRF = Map.addLayer(clasificacionRF, visClas, 'Clasificación RF', true);
var capaSVM = Map.addLayer(clasificacionSVM, visClas, 'Clasificación SVM', false);
var capaKMeans = Map.addLayer(clasificacionKMeans, visClas, 'Clasificación K-Means', false);
var capaDEM = Map.addLayer(demClip, visDEM, 'DEM', false);
var capaArea = Map.addLayer(areaEstudio, {color: 'red'}, 'Área de estudio', true);
var capaMuestras = Map.addLayer(muestrasEtiquetadas, {color: 'cyan'}, 'Muestras', true);

// -------------------------------------------------------------------------
// 9. FUNCIÓN: LEYENDA CON IMPORTANCIA POR CLASE
// -------------------------------------------------------------------------
var crearLeyendaConImportancia = function(titulo, descripcion, info) {
  var panel = ui.Panel({
    style: {
      padding: '8px 12px',
      margin: '4px 0 0 0',
      backgroundColor: '#fdfdfd',
      border: '1px solid #ccc'
    }
  });
  
  // Título
  panel.add(ui.Label({
    value: titulo,
    style: {fontWeight: 'bold', fontSize: '13px', margin: '0 0 2px 0'}
  }));
  
  // Descripción general
  panel.add(ui.Label({
    value: descripcion,
    style: {
      fontSize: '10px',
      color: '#555',
      margin: '0 0 6px 0',
      whiteSpace: 'pre-wrap',
      width: '260px'
    }
  }));
  
  // Ordenar por prioridad (1 = más importante)
  var ordenadas = info.slice().sort(function(a, b) { return a.prioridad - b.prioridad; });
  
  // Fila por cada clase con su nivel de importancia
  ordenadas.forEach(function(c) {
    var fila = ui.Panel({
      style: {margin: '0 0 6px 0'},
      layout: ui.Panel.Layout.Flow('vertical')
    });
    
    // Primera línea: caja de color + nombre + nivel
    var colorBox = ui.Label({
      style: {
        backgroundColor: c.color,
        padding: '6px',
        margin: '0 4px 0 0',
        border: '1px solid #999'
      }
    });
    var nombreLabel = ui.Label({
      value: c.nombre,
      style: {fontWeight: 'bold', fontSize: '11px', margin: '0 4px 0 0'}
    });
    var nivelLabel = ui.Label({
      value: c.nivel,
      style: {
        fontSize: '10px',
        color: c.prioridad <= 2 ? '#B8860B' : (c.prioridad <= 3 ? '#666' : '#999'),
        margin: '0'
      }
    });
    fila.add(ui.Panel({
      widgets: [colorBox, nombreLabel, nivelLabel],
      layout: ui.Panel.Layout.Flow('horizontal')
    }));
    
    // Segunda línea: significado
    fila.add(ui.Label({
      value: c.significado,
      style: {
        fontSize: '10px',
        color: '#555',
        margin: '2px 0 0 0',
        whiteSpace: 'pre-wrap',
        width: '250px'
      }
    }));
    
    // Tercera línea: uso
    fila.add(ui.Label({
      value: '▸ ' + c.uso,
      style: {
        fontSize: '10px',
        color: '#333',
        fontStyle: 'italic',
        margin: '2px 0 0 0',
        whiteSpace: 'pre-wrap',
        width: '250px'
      }
    }));
    
    panel.add(fila);
  });
  
  return panel;
};

// -------------------------------------------------------------------------
// 10. CREAR LEYENDAS
// -------------------------------------------------------------------------

var leyendaRF = crearLeyendaConImportancia(
  '🌲 Random Forest',
  'Clasificación supervisada (100 árboles). Las clases están ordenadas ' +
  'de MAYOR a MENOR importancia exploratoria.',
  infoClases
);

var leyendaSVM = crearLeyendaConImportancia(
  '📐 SVM RBF',
  'Clasificación supervisada con kernel RBF (gamma=0.5, cost=10). ' +
  'Mismo orden de importancia que Random Forest.',
  infoClases
);

var leyendaKMeans = crearLeyendaConImportancia(
  '🔵 K-Means (no supervisado)',
  'Clusters SIN etiquetas reales. NO interpretar la importancia aquí: ' +
  'los números de cluster no corresponden necesariamente a las clases.',
  infoClases.map(function(c) {
    return {
      id: c.id,
      nombre: 'Cluster ' + c.id,
      color: c.color,
      prioridad: c.prioridad,
      nivel: '??',
      significado: 'Cluster automático. Su significado real depende ' +
                   'de la correspondencia con clases geológicas (a validar).',
      uso: 'Solo como referencia exploratoria.'
    };
  })
);

// Leyenda del DEM simplificada
var leyendaDEM = (function() {
  var p = ui.Panel({style: {padding: '8px 12px', backgroundColor: '#fdfdfd',
                            border: '1px solid #ccc', margin: '4px 0 0 0'}});
  p.add(ui.Label({value: '⛰️ DEM', style: {fontWeight: 'bold', fontSize: '13px'}}));
  p.add(ui.Label({
    value: 'Modelo digital de elevación (m). Útil para contextualizar ' +
           'la cúpula y los flancos del cerro.',
    style: {fontSize: '10px', color: '#555', whiteSpace: 'pre-wrap', width: '250px'}
  }));
  [['Bajo (~3800 m)', '#0000FF'], ['Medio-bajo', '#00FF00'],
   ['Medio-alto', '#FFFF00'], ['Alto (~4800 m)', '#FF0000']].forEach(function(par) {
    var box = ui.Label({style: {backgroundColor: par[1], padding: '6px',
                                margin: '0 4px 3px 0', border: '1px solid #999'}});
    var lbl = ui.Label({value: par[0], style: {fontSize: '10px', margin: '0 0 3px 0'}});
    p.add(ui.Panel({widgets: [box, lbl], layout: ui.Panel.Layout.Flow('horizontal')}));
  });
  return p;
})();

// Leyenda de muestras
var leyendaMuestras = (function() {
  var p = ui.Panel({style: {padding: '8px 12px', backgroundColor: '#fdfdfd',
                            border: '1px solid #ccc', margin: '4px 0 0 0'}});
  p.add(ui.Label({value: '📍 Muestras', style: {fontWeight: 'bold', fontSize: '13px'}}));
  p.add(ui.Label({
    value: 'Puntos con análisis de laboratorio. Alimentan el entrenamiento ' +
           'supervisado.',
    style: {fontSize: '10px', color: '#555', whiteSpace: 'pre-wrap', width: '250px'}
  }));
  return p;
})();

// Leyenda del área
var leyendaArea = (function() {
  var p = ui.Panel({style: {padding: '8px 12px', backgroundColor: '#fdfdfd',
                            border: '1px solid #ccc', margin: '4px 0 0 0'}});
  p.add(ui.Label({value: '🔴 Área de estudio',
                  style: {fontWeight: 'bold', fontSize: '13px'}}));
  p.add(ui.Label({
    value: 'Polígono PONGO QUINUMA. Todos los productos están recortados ' +
           'a esta extensión.',
    style: {fontSize: '10px', color: '#555', whiteSpace: 'pre-wrap', width: '250px'}
  }));
  return p;
})();

// -------------------------------------------------------------------------
// 11. PANEL DE RANKING GLOBAL DE IMPORTANCIA (RESUMEN)
// -------------------------------------------------------------------------
var panelRanking = ui.Panel({
  style: {
    position: 'top-right',
    padding: '10px 12px',
    backgroundColor: 'white',
    border: '2px solid #B8860B',
    width: '280px'
  }
});

panelRanking.add(ui.Label({
  value: '🏆 RANKING DE IMPORTANCIA',
  style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 2px 0', color: '#B8860B'}
}));

panelRanking.add(ui.Label({
  value: 'Ordenado de MAYOR a MENOR interés exploratorio para Ag-Sn en Cerro Rico.',
  style: {fontSize: '10px', color: '#666', whiteSpace: 'pre-wrap',
          width: '260px', margin: '0 0 8px 0'}
}));

// Ordenar por prioridad
var infoOrdenada = infoClases.slice().sort(function(a, b) { return a.prioridad - b.prioridad; });

infoOrdenada.forEach(function(c, idx) {
  var medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣'][idx] || '▫️';
  
  var box = ui.Label({
    style: {backgroundColor: c.color, padding: '6px', margin: '0 4px 0 0',
            border: '1px solid #999'}
  });
  var nombre = ui.Label({
    value: medal + ' ' + c.nombre,
    style: {fontWeight: 'bold', fontSize: '11px', margin: '0 4px 0 0'}
  });
  var nivel = ui.Label({
    value: c.nivel,
    style: {fontSize: '10px',
            color: c.prioridad <= 2 ? '#B8860B' : '#888'}
  });
  
  var fila = ui.Panel({
    widgets: [box, nombre, nivel],
    layout: ui.Panel.Layout.Flow('horizontal'),
    style: {margin: '0 0 4px 0'}
  });
  panelRanking.add(fila);
});

panelRanking.add(ui.Label({
  value: '─────────────────',
  style: {color: '#ccc', margin: '6px 0'}
}));

panelRanking.add(ui.Label({
  value: '💡 Regla práctica:',
  style: {fontWeight: 'bold', fontSize: '11px', margin: '0 0 2px 0'}
}));

panelRanking.add(ui.Label({
  value: 'Si Random Forest y SVM coinciden en marcar grandes áreas de ' +
         'Cuarzo-sericita y Jarita, esas son las zonas prioritarias para ' +
         'exploración. Las clases Roca fresca y Vegetación deben ' +
         'descartarse del análisis.',
  style: {fontSize: '10px', color: '#444', whiteSpace: 'pre-wrap', width: '260px'}
}));

Map.add(panelRanking);

// -------------------------------------------------------------------------
// 12. PANEL DE CONTROL CON CHECKBOXES
// -------------------------------------------------------------------------
var contenedorLeyendas = ui.Panel({style: {padding: '0'}});

var crearCheckbox = function(etiqueta, capa, leyenda, activoPorDefecto) {
  var checkbox = ui.Checkbox({
    label: etiqueta,
    value: activoPorDefecto,
    style: {fontSize: '12px', margin: '2px 0'}
  });
  
  checkbox.onChange(function(valor) {
    capa.setShown(valor);
    if (valor) {
      contenedorLeyendas.add(leyenda);
    } else {
      contenedorLeyendas.remove(leyenda);
    }
  });
  
  if (activoPorDefecto) {
    contenedorLeyendas.add(leyenda);
  }
  
  return checkbox;
};

var cbRF = crearCheckbox('🌲 Random Forest', capaRF, leyendaRF, true);
var cbSVM = crearCheckbox('📐 SVM RBF', capaSVM, leyendaSVM, false);
var cbKMeans = crearCheckbox('🔵 K-Means', capaKMeans, leyendaKMeans, false);
var cbDEM = crearCheckbox('⛰️ DEM', capaDEM, leyendaDEM, false);
var cbArea = crearCheckbox('🔴 Área de estudio', capaArea, leyendaArea, true);
var cbMuestras = crearCheckbox('📍 Muestras', capaMuestras, leyendaMuestras, true);

var panelControl = ui.Panel({
  widgets: [
    ui.Label({
      value: '🎛️ CONTROL DE CAPAS',
      style: {fontWeight: 'bold', fontSize: '13px', margin: '0 0 4px 0'}
    }),
    ui.Label({
      value: 'Activa/desactiva cada producto. Su leyenda con ranking de ' +
             'importancia aparece y desaparece junto con la capa.',
      style: {fontSize: '10px', color: '#666', whiteSpace: 'pre-wrap',
              width: '270px', margin: '0 0 6px 0'}
    }),
    cbRF,
    cbSVM,
    cbKMeans,
    cbDEM,
    cbArea,
    cbMuestras,
    ui.Label({value: '─────────────────', style: {color: '#ccc', margin: '6px 0 4px 0'}}),
    ui.Label({
      value: '📖 LEYENDAS (con importancia)',
      style: {fontWeight: 'bold', fontSize: '12px', margin: '0 0 4px 0'}
    }),
    contenedorLeyendas
  ],
  style: {
    position: 'bottom-left',
    padding: '10px',
    backgroundColor: 'white',
    border: '1px solid #999',
    width: '290px',
    maxHeight: '550px'
  }
});

Map.add(panelControl);

// -------------------------------------------------------------------------
// 13. MÉTRICAS Y CONTEXTO
// -------------------------------------------------------------------------
print('=====================================================');
print('=== CONTEXTO GEOLÓGICO Y JERARQUÍA DE CLASES ===');
print('=====================================================');
print('Cerro Rico — depósito Ag-Sn polimetálico hidrotermal mioceno.');
print('');
print('Orden de importancia exploratoria (de MAYOR a MENOR):');
infoOrdenada.forEach(function(c) {
  print(c.prioridad + '. ' + c.nombre + ' — ' + c.nivel);
  print('     ' + c.significado);
});