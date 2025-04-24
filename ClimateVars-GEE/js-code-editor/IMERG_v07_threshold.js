// === PARÂMETROS ===
var regionFeatureCollection = "projects/ee-leoippef/assets/BR_UF_2021"; // Nome do FeatureCollection com a região
var regionFilterField = "SIGLA";   // Nome do campo a ser filtrado
var regionFilterValue = "MG";        // Valor do campo a ser filtrado
var startDate = "2024-01-01";           // Data de início
var endDate = "2024-12-31";             // Data de fim
var mode = "mensal";                    // ou "diario"
var folderName = "GPM_MG_BaixaPrecip";  // Pasta no Drive
var maxPrecip = 100;                     // Valor máximo de precipitação a exportar (em mm)

// === SCRIPT ===
var region = ee.FeatureCollection(regionFeatureCollection)
              .filter(ee.Filter.eq(regionFilterField, regionFilterValue));
Map.centerObject(region, 6);
Map.addLayer(region, {}, 'Região');

var start = ee.Date(startDate);
var end = ee.Date(endDate);

var timeUnit = (mode === 'mensal') ? 'month' : 'day';
var formatStr = (mode === 'mensal') ? 'YYYY-MM' : 'YYYY-MM-dd';

var count = end.difference(start, timeUnit).round();
var dateList = ee.List.sequence(0, count.subtract(1)).map(function(offset) {
  return start.advance(offset, timeUnit);
});

var images = dateList.map(function(date) {
  date = ee.Date(date);
  var img = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
    .filterDate(date, date.advance(1, timeUnit))
    .select('precipitation')
    .sum()
    .clip(region)
    .set('date_export', date.format(formatStr));
  return img;
});

var imageList = ee.ImageCollection(images).toList(count);
var n = count.getInfo();

// ⚠️ EXPORTAÇÃO FILTRANDO PRECIPITAÇÃO MÁXIMA
for (var i = 0; i < n; i++) {
  var img = ee.Image(imageList.get(i));
  var dateStr = img.get('date_export').getInfo();

  // Calcular média da imagem para decidir se exporta
  var meanPrecip = img.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region.geometry(),
    scale: 10000,
    maxPixels: 1e13
  }).get('precipitation');

  // Só exporta se a média for menor que o limite
  if (meanPrecip !== null && meanPrecip.getInfo() < maxPrecip) {
    Export.image.toDrive({
      image: img,
      description: 'GPM_' + regionFilterValue + '_' + dateStr,
      folder: folderName,
      fileNamePrefix: 'GPM_' + dateStr,
      region: region.geometry(),
      scale: 10000,
      maxPixels: 1e13
    });
  }
}
