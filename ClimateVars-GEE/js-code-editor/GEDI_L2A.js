// === PARÂMETROS ===
var regionFeatureCollection = "ASSET"; // FeatureCollection com estados
var regionFilterField = "FIELD_NAME"; // Campo a ser filtrado
var regionFilterValue = "VALUE"; // Valor do campo (ex: MG)
var folderName = "GEDI_Export"; // Pasta no Google Drive
var bandName = "rh98"; // Banda de interesse
var descPrefix = "GEDI_" + bandName + "_MEAN_";
var startDate = "2019-03-25"; // Data inicial 2019-03-25T00:00:00Z - min
var endDate = "2024-11-01"; // Data final 2024-11-01T08:00:00Z - max

// === DEFINIR REGIÃO ===
var region = ee.FeatureCollection(regionFeatureCollection)
              .filter(ee.Filter.eq(regionFilterField, regionFilterValue));
Map.centerObject(region, 7);
Map.addLayer(region, {}, 'Região');

// === APLICAR MÁSCARAS E FILTRAR COLEÇÃO ===
var dataset = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
  .filterDate(startDate, endDate)
  .map(function(im) {
    // Primeiro aplica as máscaras, depois seleciona apenas a banda
    return im.updateMask(im.select('quality_flag').eq(1))
             .updateMask(im.select('degrade_flag').eq(0))
             .select(bandName)
             .clip(region);
  });

// === CÁLCULO DA MÉDIA TEMPORAL ===
var rh98Mean = dataset.mean().rename(bandName);

// === VISUALIZAÇÃO ===
var visParams = {
  min: 1,
  max: 60,
  palette: ['darkred', 'red', 'orange', 'green', 'darkgreen']
};
Map.addLayer(rh98Mean, visParams, 'rh98 Mean');

// === EXPORTAÇÃO PARA DRIVE ===
Export.image.toDrive({
  image: rh98Mean,
  description: descPrefix + regionFilterValue + "_" + startDate + "_TO_" + endDate,
  folder: folderName,
  fileNamePrefix: descPrefix + regionFilterValue,
  region: region.geometry(),
  scale: 25,
  maxPixels: 1e13
});
