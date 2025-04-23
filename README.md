# 🌧️🌲 Markowitz Climático — Fronteira de Eficiência com Rasters

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
