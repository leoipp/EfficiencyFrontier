import ee

# === PARÂMETROS ===
region_feature_collection = "projects/ee-leoippef/assets/BR_UF_2021"  # Nome do seu asset
region_filter_field = "SIGLA"   # Campo para filtrar
region_filter_value = "MG"         # Valor do filtro
start_date = "2015-01-01"
end_date = "2024-12-31"
mode = "mensal"  # ou "diario"
folder_name = "GPM_MG_Export_2015_2014"  # Pasta no Google Drive

# === AUTENTICAR E INICIALIZAR ===
ee.Authenticate()
ee.Initialize(project='ee-leoippef')

# === SCRIPT ===
# Carrega e filtra a região
region = ee.FeatureCollection(region_feature_collection) \
           .filter(ee.Filter.eq(region_filter_field, region_filter_value))

# Define datas
start = ee.Date(start_date)
end = ee.Date(end_date)

# Define unidade de tempo
time_unit = 'month' if mode == 'mensal' else 'day'
format_str = 'YYYY-MM' if mode == 'mensal' else 'YYYY-MM-dd'

# Gera a lista de datas
count = end.difference(start, time_unit).round()
date_list = ee.List.sequence(0, count.subtract(1)).map(lambda offset: start.advance(offset, time_unit))

# Cria coleção de imagens
def create_image(date):
    date = ee.Date(date)
    img = ee.ImageCollection('NASA/GPM_L3/IMERG_V07') \
            .filterDate(date, date.advance(1, time_unit)) \
            .select('precipitation') \
            .sum() \
            .clip(region) \
            .set('date_export', date.format(format_str))
    return img

images = date_list.map(create_image)
image_list = ee.ImageCollection(images).toList(count)

# ⚠️ LOOP DE EXPORTAÇÃO
n = count.getInfo()
for i in range(n):
    img = ee.Image(image_list.get(i))
    date_str = img.get('date_export').getInfo()

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f'GPM_{region_filter_value}_{date_str}',
        folder=folder_name,
        fileNamePrefix=f'GPM_{date_str}',
        region=region.geometry(),
        scale=10000,
        maxPixels=1e13
    )
    task.start()
    print(f"Export task started for date {date_str}")
