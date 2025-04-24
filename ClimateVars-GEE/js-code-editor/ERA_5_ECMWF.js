// === PARÂMETROS ===
var regionFeatureCollection = "ASSET"; // Nome do FeatureCollection com a região
var regionFilterField = "FIELD_NAME";   // Nome do campo a ser filtrado
var regionFilterValue = "VALUE";        // Valor do campo a ser filtrado
var startDate = "2020-01-01";
var endDate = "2020-12-31";
var mode = "mensal"; // ou "diario"
var band = "temperature_2m"; // Nome da banda de analise
var bandNewName = "temperature_K"; // Novo nome de banda de analise
var folderName = "ECMWF_Temp_Export"; // Pasta no Drive
var descp = "ECMWF_temp_"; // Raster description initial

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
  var collection = (mode === 'mensal') ?
    ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR') :
    ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
  var img = collection
    .filterDate(date, date.advance(1, timeUnit))
    .select(band)
    .mean()
    .rename(bandNewName)
    .clip(region)
    .set('date_export', date.format(formatStr));

  return img;
});

var imageList = ee.ImageCollection(images).toList(count);

// === EXPORTAÇÃO PARA O DRIVE ===
var n = count.getInfo();
for (var i = 0; i < n; i++) {
  var img = ee.Image(imageList.get(i));
  var dateStr = img.get('date_export').getInfo();

  Export.image.toDrive({
    image: img,
    description: descp + regionFilterValue + '_' + dateStr,
    folder: folderName,
    fileNamePrefix: descp + dateStr,
    region: region.geometry(),
    scale: 10000,
    maxPixels: 1e13
  });
}
