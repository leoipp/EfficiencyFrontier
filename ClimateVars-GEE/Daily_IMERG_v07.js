// 1. Importar a geometria de Minas Gerais
var estados = ee.FeatureCollection("projects/ee-leoippef/assets/BR_UF_2021");
var mg = estados.filter(ee.Filter.eq('SIGLA', 'MG'));
Map.centerObject(mg, 6);
Map.addLayer(mg, {}, 'Minas Gerais');

// 2. Definir o intervalo de datas
var start = ee.Date('2010-01-01');
var end = ee.Date('2024-12-31');

// 3. Criar lista de datas diárias
var nDays = end.difference(start, 'day');
var dates = ee.List.sequence(0, nDays.subtract(1)).map(function(dayOffset) {
  return start.advance(dayOffset, 'day');
});

// 4. Gerar coleção de imagens com data como metadado
var images = dates.map(function(date) {
  date = ee.Date(date);
  var img = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
    .filterDate(date, date.advance(1, 'day'))
    .select('precipitation')
    .sum()
    .clip(mg)
    .set('date_export', date.format('YYYY-MM-dd'));  // salva a data como metadado
  return img;
});

var imageList = ee.ImageCollection(images).toList(nDays);

// 5. Loop client-side para exportar
for (var i = 0; i < nDays.getInfo(); i++) {
  var img = ee.Image(imageList.get(i));
  var dateStr = img.get('date_export').getInfo();

  Export.image.toDrive({
    image: img,
    description: 'GPM_MG_' + dateStr,
    folder: 'GPM_MG_Setembro2019',
    fileNamePrefix: 'GPM_' + dateStr,
    region: mg.geometry(),
    scale: 10000,
    maxPixels: 1e13
  });
}
