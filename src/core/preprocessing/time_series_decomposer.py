from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesDecomposer:

    # def decompose_time_series():

    def is_stationary(self, df, target_col, significance_level=0.05, verbose=False):
        """
        Verifies if a time series is stationary using the Augmented Dickey-Fuller test.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the time series data.
            target_col (str): The column name of the time series.
            significance_level (float): The significance level for the test (default is 0.05).
            verbose (bool): If True, print the test results (default is False).

        Returns:
            bool: True if the time series is stationary, False otherwise.
        """
        adf_result = adfuller(df[target_col])
        p_value = adf_result[1]

        if verbose:
            print('ADF Statistic: %f' % adf_result[0])
            print('p-value: %f' % p_value)
            print('Critical Values:')
            for key, value in adf_result[4].items():
                print('\t%s: %.3f' % (key, value))

        # Check if the p-value is less than the significance level
        is_stationary = p_value < significance_level

        return is_stationary

    def make_series_stationary(self, df, target_col):
        """
        Makes a time series stationary by applying differencing.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the time series data.
            target_col (str): The column name of the time series.

        Returns:
            pd.DataFrame: A new DataFrame with the stationary time series.
        """
        # Copy the original DataFrame to avoid modifying the input DataFrame

        is_stationary = False
        df_stationary = df.copy()
        order_diff_used = None
        for order_d in [1, 2, 3]:
            if is_stationary:
                break
            # Apply differencing to make the series stationary
            order_diff_used = order_d
            new_target_col = f"d{order_d}_'{target_col}"
            df_stationary[new_target_col] = df_stationary[target_col].diff().fillna(
                0)
            # Drop the original target_col since it's no longer needed
            df_stationary.drop(columns=target_col, inplace=True)

            is_stationary = self.is_stationary(df_stationary, new_target_col)

        return df_stationary, order_diff_used, new_target_col

    def decompose_time_series(self, df, target_column):
        if target_column not in df.columns:
            raise ValueError(
                f"La columna objetivo '{target_column}' no existe en el DataFrame.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "El índice del DataFrame no es de tipo DatetimeIndex. Asegúrate de que tus fechas estén en el índice.")

        # Realizar la descomposición utilizando STL (Seasonal and Trend decomposition using Loess)
        # Si tus datos son diarios y esperas que haya patrones estacionales que se repiten cada 7 días (semanalmente), podrías usar seasonal=7.
        # Ajusta el valor 'seasonal' según la periodicidad de los datos
        decomposer = STL(df[target_column], seasonal=7)
        decomposition_result = decomposer.fit()

        return decomposition_result

    def visualize_decomposition(self, decomposition_result, output_file=None):
        # Visualizar los componentes de la descomposición
        fig, axes = plt.subplots(4, figsize=(14, 10))

        axes[0].plot(decomposition_result.trend, label='Tendencia')
        axes[0].set_title('Tendencia')

        axes[1].plot(decomposition_result.seasonal, label='Estacionalidad')
        axes[1].set_title('Estacionalidad')

        axes[2].plot(decomposition_result.resid, label='Residuo')
        axes[2].set_title('Residuo')

        axes[3].plot(decomposition_result.observed, label='Observado')
        axes[3].set_title('Observado')

        for ax in axes:
            ax.legend()

        # Mostrar el gráfico o guardar la figura en el archivo
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def seasonal_plot(self, X, y, period, freq, ax=None, output_file=None):
        """
        Create a seasonal plot to visualize patterns in time series data.

        Parameters:
        - X: pandas DataFrame. The input data containing the time series.
        - y: str. The column name of the dependent variable to plot.
        - period: str. The name of the column representing the seasonal period.
        - freq: str. The name of the column representing the time frequency.
        - ax: matplotlib Axes, optional. The axes on which to plot.
        - output_file: str, optional. The file path to save the plot image (e.g., 'plot.png').

        Example:
        >>> X = pd.read_csv('data.csv')  # Load your data
        >>> your_instance = YourClass()  # Create an instance of your class
        >>> your_instance.seasonal_plot(X, y='sales', period='month', freq='year')

        This function creates a seasonal plot of the specified time series data using Seaborn.
        It visualizes patterns across different periods and frequencies.

        If an 'ax' parameter is provided, the plot will be drawn on the specified axes.
        If 'output_file' is provided, the plot will be saved as an image at the specified path.
        Otherwise, the plot will be displayed using plt.show().
        """
        if ax is None:
            _, ax = plt.subplots()

        palette = sns.color_palette("husl", n_colors=X[period].nunique())

        ax = sns.lineplot(
            x=freq,
            y=y,
            hue=period,
            data=X,
            ci=False,
            ax=ax,
            palette=palette,
            legend=False,
        )

        ax.set_title(f"Seasonal Plot ({period}/{freq})")

        for line, name in zip(ax.lines, X[period].unique()):
            y_ = line.get_ydata()[-1]
            ax.annotate(
                name,
                xy=(1, y_),
                xytext=(6, 0),
                color=line.get_color(),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                size=14,
                va="center",
            )

        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
