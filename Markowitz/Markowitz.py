import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt


class Markowitz:
    """
    A classe Markowitz é o coração do código. Ela tem como objetivo simular a fronteira de eficiência climática com
    base em rasters de precipitação. A analogia seria algo como um analista de investimentos que deseja analisar os
    dados climáticos como se fossem ativos financeiros.
    """
    __version__ = "0.1b"
    __author__ = "<Leonardo Ippolito Rodrigues>"
    def __init__(self, raster_path_pattern, target_raster=None, num_pixels=None, seed=42):
        """
        Inicializa a análise de Markowitz sobre rasters.
            :param raster_path_pattern: Padrão para arquivos, ex: 'data/precip_2019-09-*.tif'
            :param target_raster: Raster de retorno (produção) como argumento opcional
            :param num_pixels: Número de pixels a amostrar
            :param seed: Semente para replicabilidade
        """
        self.raster_path_pattern = raster_path_pattern
        self.target_raster_path = target_raster
        self.num_pixels = num_pixels
        self.seed = seed
        self.weights_list = []
        self.target_values = None
        self.stack = None
        self.series = None
        self.mean_precip = None
        self.std_precip = None
        self.cov_matrix = None
        self.results = None
        self.coords = None

    def load_stack(self):
        """
        O metodo carrega todos os rasters em um array 3D.Isso pode ser comparado a coletar dados financeiros de
        diferentes ativos ao longo de vários dias. Aqui, os ativos são os valores de precipitação diários em várias
        regiões. Basicamente, amostra N pixels válidos e extrai séries temporais.
            Como funciona:
                * O código lê todos os arquivos TIFF que correspondem ao padrão raster_path_pattern.
                * Ele os empilha em uma matriz 3D: Cada camada da matriz será um raster para um dia, e as linhas e
                colunas representam os diferentes pixels (como diferentes ativos financeiros em diferentes datas).
        """
        files = sorted(glob.glob(self.raster_path_pattern))
        stack = [rasterio.open(f).read(1) for f in files]
        self.stack = np.array(stack)
        print(f"Stack carregada: {self.stack.shape}")
        print("Total de NaNs no stack:", np.isnan(self.stack).sum())

    def sample_pixels(self, threshold: float=0.0, data_percent_tolerance: float=0.7):
        """
        O objetivo aqui é selecionar um conjunto de pixels aleatórios a partir do stack de precipitação para análise.
        Aqui, estamos basicamente escolhendo alguns "ativos" (ou pixels de precipitação) para observar como eles
        variam ao longo do tempo.
            Como funciona:
                 * O código verifica quais pixels possuem valores válidos (precipitação maior que 0).
                 * Em seguida, amostra número de pixels aleatórios, como se estivéssemos escolhendo ativos financeiros
                 para construir um portfólio.
                 * A série temporal de precipitação desses pixels selecionados será nossa série temporal de retornos (
                 simulando o desempenho dos ativos ao longo do tempo).
        """
        if self.stack is None:
            raise ValueError("Stack não carregada. Use .load_stack() antes.")

        self.stack = np.nan_to_num(self.stack)  # remove NaNs substituindo por 0
        self.stack[self.stack < threshold] = 0  # aplica threshold

        # Máscara para pixels válidos em pelo menos 70% das datas
        valid_ratio = np.mean(self.stack > 0, axis=0)  # (98, 126)
        valid_mask = valid_ratio >= data_percent_tolerance

        print("Pixels válidos após nova máscara:", np.sum(valid_mask))
        ys, xs = np.where(valid_mask)
        coords = list(zip(ys, xs))

        if self.num_pixels is None:
            self.coords = coords
            print(f"{len(coords)} pixels amostrados com sucesso.")
        else:
            if self.num_pixels > len(coords):
                raise ValueError(f"Você pediu {self.num_pixels} pixels, mas só existem {len(coords)} válidos.")
            np.random.seed(self.seed)
            sampled = np.random.choice(len(coords), self.num_pixels, replace=False)
            self.coords = [coords[i] for i in sampled]
            print(f"{self.num_pixels} pixels amostrados com sucesso.")

        self.series = np.array([self.stack[:, y, x] for y, x in self.coords])

        # Carregar raster de retorno real (ex: produção)
        if self.target_raster_path:
            with rasterio.open(self.target_raster_path) as src:
                target_data = src.read(1)
                self.target_values = np.array([target_data[y, x] for y, x in self.coords])
                print("Valores de retorno (produção) extraídos com sucesso.")

    def calculate_statistics(self):
        """
        Agora, vamos calcular médias, desvios padrão e matriz de covariância das séries temporais de precipitação
        dos pixels amostrados. Essas estatísticas são essenciais para entender a performance histórica e o risco de
        cada pixel.
            Como funciona:
                * Média de Precipitação: É a média de retornos para cada pixel (ativo).
                * Desvio Padrão: Medimos a volatilidade de cada pixel ao longo do tempo, ou seja, a incerteza associada
                ao seu  comportamento (o risco).
                * Matriz de Covariância: A covariância entre os diferentes pixels (ou ativos) mostra como eles se
                comportam em  conjunto ao longo do tempo, ajudando a entender se eles se movem juntos (correlação).
        """
        if self.series is None:
            raise ValueError("Pixels não amostrados. Use .sample_pixels() antes.")

        self.mean_precip = self.series.mean(axis=1)
        self.std_precip = self.series.std(axis=1)
        self.cov_matrix = np.cov(self.series)
        print("Estatísticas climáticas calculadas.")

    def simulate_portfolios(self, num_portfolios: int=1000) -> None:
        """
        Agora, o código vai simular a composição de diversos portfólios climáticos. Cada portfólio será uma
        combinação de pesos diferentes atribuídos a cada pixel (ativo). A partir disso, vamos calcular a média de
        precipitação (retorno) e o risco (desvio padrão) associado a cada combinação.
            Como funciona:
                * Pesos aleatórios são atribuídos a cada pixel.
                * Para cada combinação de pesos, é calculado o retorno esperado e o risco do portfólio, assim como o
                Índice de Sharpe.
                    * Retorno: Média ponderada das precipitações.
                    * Risco: Desvio padrão ponderado usando a matriz de covariância.
                    * Índice de Sharpe: Calculado dividindo o retorno pelo risco, ajudando a determinar qual
                    combinação tem o melhor retorno ajustado ao risco.
        :param num_portfolios: Numero de portifolios a serem simulados
        :return:
        """
        if self.cov_matrix is None:
            raise ValueError("Estatísticas não calculadas. Use .calculate_statistics() antes.")

        results = np.zeros((3, num_portfolios))
        n = len(self.mean_precip)

        for i in range(num_portfolios):
            weights = np.random.random(n)
            weights /= np.sum(weights)
            self.weights_list.append(weights)

            retorno = np.dot(weights, self.mean_precip)
            risco = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = retorno / risco

            results[0, i] = risco
            results[1, i] = retorno
            results[2, i] = sharpe

        self.results = results
        print(f"{num_portfolios} portfolios simulados.")

    def evaluate_against_target(self):
        """
        Calcula o retorno real dos portfólios usando o raster de retorno
        """
        if self.target_values is None or self.weights_list is None:
            raise ValueError("Valores de retorno reais ou pesos não estão disponíveis.")

        real_returns = []
        for weights in self.weights_list:
            retorno_real = np.dot(weights, self.target_values)
            real_returns.append(retorno_real)

        return np.array(real_returns)

    def plot_frontier(self):
        """Plota a Fronteira de Eficiência"""
        if self.results is None:
            raise ValueError("Resultados não simulados. Use .simulate_portfolios() antes.")

        risco, retorno, sharpe = self.results
        plt.figure(figsize=(10, 6))
        plt.scatter(risco, retorno, c=sharpe, cmap='plasma', s=10)
        plt.xlabel('Risco (Desvio padrão)')
        plt.ylabel('Retorno (Precipitação média)')
        plt.title('Fronteira de Eficiência Climática')
        plt.colorbar(label='Índice Sharpe Climático')
        plt.grid(linestyle='--')
        plt.show()

    def plot_real_frontier(self):
        """
        Plota Risco x Retorno real baseado no raster de produção
        """
        if self.results is None:
            raise ValueError("Resultados não simulados.")
        real_returns = self.evaluate_against_target()
        riscos = self.results[0]

        plt.figure(figsize=(10, 6))
        plt.scatter(riscos, real_returns, c=real_returns, cmap='viridis', s=10)
        plt.xlabel('Risco (Desvio padrão)')
        plt.ylabel('Retorno real (Produção)')
        plt.title('Fronteira baseada em Produção')
        plt.colorbar(label='Produção estimada')
        plt.grid(linestyle='--')
        plt.show()

    def get_high_sharpe_precip(self, threshold: float=1.0) -> None:
        """
        Retorna a precipitação real ponderada dos portfólios com Sharpe acima do threshold.
        :param threshold: valor mínimo de Sharpe
        :return: lista de arrays de precipitação ponderada
        """
        if self.results is None or self.weights_list is None:
            raise ValueError("Portfólios não simulados ainda.")

        riscos, retornos, sharpes = self.results
        high_sharpe_indices = np.where(sharpes >= threshold)[0]

        if len(high_sharpe_indices) == 0:
            print("Nenhum portfólio com Sharpe acima do threshold.")
            return []

        selected_precips = []
        for idx in high_sharpe_indices:
            weights = self.weights_list[idx]
            # Combina a série temporal real com os pesos (precipitação ponderada ao longo do tempo)
            combined = np.dot(weights, self.series)
            selected_precips.append(combined)

        print(f"{len(selected_precips)} portfólios selecionados com Sharpe >= {threshold}")
        return selected_precips


mk = Markowitz('C:/Users/c0010261/Scripts/EfficiencyFrontier/Example/GPM_2019-09-0*.tif')

mk.load_stack()
mk.sample_pixels()
mk.calculate_statistics()
mk.simulate_portfolios()
mk.plot_frontier()

"""
Variáveis climáticas com avaliação de retorno por pixel dentre as séries temporais:
    mk.load_stack()
    mk.sample_pixels()
    mk.calculate_statistics()
    mk.simulate_portfolios()
    mk.plot_frontier()
    mk.get_high_sharpe_precip(1.5)
"""
"""
Variáveis climaticas com avaliação de retorno sobre outra variavel Ex. Produção volumétrica:
    mk.load_stack()
    mk.sample_pixels()
    mk.calculate_statistics()
    mk.simulate_portfolios()
    mk.plot_real_frontier()
"""