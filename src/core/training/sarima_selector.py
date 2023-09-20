import itertools
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMASelector:
    def __init__(self, data, p_range, d_range, q_range, m_values, trend='c', criterion='aic'):
        self.data = data
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.m_values = m_values
        self.trend = trend
        self.criterion = criterion

    def _get_aic_or_bic(self, p, d, q, m):
        model = SARIMAX(self.data, order=(p, d, q), seasonal_order=(p, d, q, m), trend=self.trend)
        result = model.fit()
        if self.criterion == 'bic':
            return result.bic
        else:
            return result.aic

    def find_optimal_params(self):
        param_combinations = list(itertools.product(self.p_range, self.d_range, self.q_range, self.m_values))
        results = []

        for params in param_combinations:
            p, d, q, m = params
            criterion_value = self._get_aic_or_bic(p, d, q, m)
            results.append((params, criterion_value))

        results.sort(key=lambda x: x[1])
        optimal_params, optimal_criterion_value = results[0]
        optimal_P, optimal_D, optimal_Q, optimal_m = optimal_params

        return optimal_P, optimal_D, optimal_Q, optimal_m

# # Ejemplo de uso:
# # Cargar los datos
# data = pd.read_csv("tu_archivo.csv")  # Reemplaza "tu_archivo.csv" por el nombre de tu archivo de datos

# # Definir el rango de valores para P, D, Q y m
# p_range = range(0, 3)
# d_range = range(0, 2)
# q_range = range(0, 3)
# m_values = [12]

# # Crear el selector
# selector = SARIMASelector(data['num_sold'], p_range, d_range, q_range, m_values)

# # Encontrar los valores óptimos de P, D, Q y m utilizando el criterio AIC
# optimal_P, optimal_D, optimal_Q, optimal_m = selector.find_optimal_params()

# print(f"Valores óptimos: P = {optimal_P}, D = {optimal_D}, Q = {optimal_Q}, m = {optimal_m}")
