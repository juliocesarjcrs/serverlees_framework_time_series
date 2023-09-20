import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class ARIMASelector:
    def __init__(self, data, p_range, q_range):
        """
        Inicializa la clase ARIMASelector.

        Parámetros:
            data (pandas.Series): Serie de tiempo original.
            p_range (range): Rango de valores para el parámetro "p".
            q_range (range): Rango de valores para el parámetro "q".
        """
        self.data = data
        self.p_range = p_range
        self.q_range = q_range

    def find_optimal_order(self, order_diff):
        """
        Encuentra los valores óptimos de "p" y "q" utilizando AIC.

        Retorna:
            best_p (int): Mejor valor de "p".
            best_q (int): Mejor valor de "q".
        """
        best_aic = np.inf
        best_p = 0
        best_q = 0

        for p in self.p_range:
            for q in self.q_range:
                try:
                    model = ARIMA(self.data, order=(p, order_diff, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
                        best_q = q

                except:
                    continue

        return best_p, best_q


