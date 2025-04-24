// === PARÂMETROS ===
var regionFeatureCollection = "ASSET"; // Nome do FeatureCollection com a região
var regionFilterField = "FIELD_NAME";   // Nome do campo a ser filtrado
var regionFilterValue = "VALUE";        // Valor do campo a ser filtrado
var folderName = "SRTM_Export";         // Pasta no Google Drive

// === SCRIPT ===
var region = ee.FeatureCollection(regionFeatureCollection)
              .filter(ee.Filter.eq(regionFilterField, regionFilterValue));
Map.centerObject(region, 8);
Map.addLayer(region, {}, 'Região');

// Carrega dados SRTM e recorta para a região
var srtm = ee.Image("USGS/SRTMGL1_003").clip(region);

// Calcula declividade
var slope = ee.Terrain.slope(srtm);

// Exporta elevação
Export.image.toDrive({
  image: srtm.rename('elevation'),
  description: 'SRTM_Elevation_' + regionFilterValue,
  folder: folderName,
  fileNamePrefix: 'SRTM_Elevation_' + regionFilterValue,
  region: region.geometry(),
  scale: 90,
  maxPixels: 1e13
});

// Exporta declividade
Export.image.toDrive({
  image: slope.rename('slope'),
  description: 'SRTM_Slope_' + regionFilterValue,
  folder: folderName,
  fileNamePrefix: 'SRTM_Slope_' + regionFilterValue,
  region: region.geometry(),
  scale: 90,
  maxPixels: 1e13
});
