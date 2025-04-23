# 🌧️🌲 Fronteira de Eficiência

Este projeto adapta os princípios do modelo de portfólios de Markowitz para análise climática e ambiental baseada em dados raster (i.e. precipitação, temperatura, umidade e outros). A ideia é tratar pixels como "ativos" e analisar seu comportamento ao longo do tempo — simulando portfólios e encontrando composições com melhor relação retorno/risco.

Agora com suporte para **variáveis de retorno real**, permitindo análises ainda mais poderosas! 🌱

---

## 📦 Funcionalidades

- 📁 Carregamento de uma pilha temporal de rasters (ex: precipitação diária)
- 🎲 Amostragem aleatória ou total de pixels válidos
- 📊 Cálculo de estatísticas: média, desvio padrão, covariância
- 🧮 Simulação de milhares de portfólios climáticos
- 🔥 Cálculo do Índice de Sharpe Climático
- 🧑‍🌾 Comparação com raster de retorno real (ex: produção agrícola/florestal)
- 📈 Visualizações da fronteira de eficiência tradicional ou baseada na produção

---

## 🚀 Como Usar

### 1. Instale as dependências

```bash
pip install numpy rasterio matplotlib
```

### 2. Estrutura esperada
```bash
📂 data/
├── clim_var_2019-09-01.tif
├── clim_var_2019-09-02.tif
├── ...
├── target_var.tif  # (opcional)
```

### 3. Código de exemplo

```bash
from markowitz import Markowitz

mk = Markowitz(
    raster_path_pattern='data/precip_2019-09-*.tif',
    num_pixels=500,
    target_raster='data/producao.tif'  # Opcional
)

mk.load_stack()
mk.sample_pixels()
mk.calculate_statistics()
mk.simulate_portfolios()
mk.plot_frontier()           # Risco x Retorno (precipitação)
mk.plot_real_frontier()      # Risco x Retorno real (produção)
```

---

## 🧠 Explicação

* **Rasters climáticos** = ativos financeiros
* **Pixels** = cada ativo individual
* **Série temporal de pixels** = retornos históricos
* **Índice de Sharpe** = eficiência climática
* **Raster target** = retorno observado do "mundo real"

--

## 📚 Base Teórica

Inspirado no modelo clássico de Harry Markowitz
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
