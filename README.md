# 🌧️☀️🌲 Efficiency Frontier

Efficiency Frontier for Environmental Modeling is a Python-based tool that adapts Modern Portfolio Theory (Markowitz) to environmental and climatic datasets. It models spatial and temporal variability by treating environmental pixels (e.g., precipitation from raster time series) as "climate assets", simulating portfolios to construct a climatic efficiency frontier.

This frontier highlights combinations of pixels with the best trade-offs between average climatic return (e.g., mean rainfall) and risk (temporal variability), using a climate Sharpe index. Optionally, the model can be anchored to real-world outcomes such as crop yield or forest productivity, using a target raster to evaluate how simulated portfolios perform in practice.

In addition, the project introduces a novel module for extracting and analyzing the temporal maximum values across the raster time series. These maxima are essential for identifying climate extremes (e.g., droughts, heatwaves, storm peaks), and can serve as an additional layer for risk assessment or feature selection.

---

## 📦 Functionalities

- 📁 Loading a temporal stack of rasters (e.g., daily precipitation)
- 🎲 Random or full sampling of valid pixels
- 📊 Calculation of statistics: mean, standard deviation, covariance
- 🧮 Simulation of thousands of climate portfolios
- 🔥 Computation of the Climate Sharpe Index
- 🧑‍🌾 Comparison with real return raster (e.g., agricultural or forest productivity)
- 📈 Visualizations of the efficiency frontier (traditional or return-based)

---

## 🚀 How to use?

### 1. Dependencies

```bash
pip install numpy rasterio matplotlib
```

### 2. Expected structure
```bash
📂 data/
├── clim_var_2019-09-01.tif
├── clim_var_2019-09-02.tif
├── ...
├── target_var.tif  # Optional
```

### 3. Code example

```bash
from markowitz import Markowitz

mk = Markowitz(
    raster_path_pattern='data/precip_2019-09-*.tif',
    num_pixels=500,
    target_raster='data/producao.tif'  # Optional
)

mk.load_stack()
mk.sample_pixels()
mk.calculate_statistics(threshold=0, data_percent_tolerance=0.7)
mk.simulate_portfolios(num_portfolios=1000)
mk.plot_frontier()           # Risk x Return (clim_var)
mk.plot_real_frontier()      # Risk x Real Return (target_var)

mk.get_high_sharpe_precip(threshold=1.0)      # Retrieve raster data for modelling
```

---

## 🧠 Explanation

* **Climate rasters** = financial assets
* **Pixels** = each individual asset
* **Pixel time series** = historical returns
* **Sharpe Index** = climate efficiency
* **Target raster** = observed "real-world" return

---

## 📚 Theoretical Foundation

Inspired by the classic model of Harry Markowitz
- [Markowitz, H. (1952). Portfolio Selection. Journal of Finance.](https://www.researchgate.net/publication/228051028_Portfolio_Selection)
