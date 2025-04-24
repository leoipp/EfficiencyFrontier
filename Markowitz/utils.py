import numpy as np


def validate_array_dtype(array: np.ndarray, valid_dtypes: list) -> np.ndarray:
    """
    Valida e converte o tipo de dado de um array para um tipo compatível.

    :param array: Array numpy a ser validado.
    :param valid_dtypes: Lista de tipos de dados válidos.
    :return: Array convertido para um tipo de dado válido, se necessário.
    """
    if array.dtype not in valid_dtypes:
        return array.astype(np.float32)
    return array


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normaliza os pesos para que a soma seja igual a 1.

    :param weights: Array de pesos.
    :return: Array de pesos normalizados.
    """
    return weights / np.sum(weights)


def calculate_sharpe_ratio(return_mean: float, risk: float) -> float:
    """
    Calcula o índice de Sharpe dado o retorno médio e o risco.

    :param return_mean: Retorno médio do portfólio.
    :param risk: Risco (desvio padrão) do portfólio.
    :return: Índice de Sharpe.
    """
    if risk == 0:
        return 0
    return return_mean / risk
