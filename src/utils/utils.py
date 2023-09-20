import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import joblib
import pandas as pd
import locale
# metrics error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from pmdarima.metrics import smape
# handle time
import locale
locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')
#  plot


class Utils:
    def load_from_csv(self, path, date_col_name, freq=None):
        """
        Load data from a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file.
        date_col_name : str, optional
            The name of the column containing the dates in the CSV file.
            Default is 'date'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the data from the CSV file with dates
            properly formatted.
        """
        df = pd.read_csv(path, parse_dates=[date_col_name])
        df[date_col_name] = pd.to_datetime(df[date_col_name])
        df[date_col_name] = df[date_col_name].dt.tz_localize(None)

        # Configurar el índice del DataFrame si no es de tipo DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index(date_col_name, inplace=True)
        if freq:
            df.index.freq = freq
        return df

    def model_export(self,  cls, score):
        directory = './models'
        isdir = os.path.isdir(directory)
        # si no existe crea el directorio
        print('isdir', isdir)
        if not isdir:
            os.mkdir(directory)
        isdir2 = os.path.isdir(directory)
        print('isdir2', isdir2)
        score_str = str(score).replace(".", "_")
        file_name = f'best_model_{score_str}.pkl'
        joblib.dump(cls, f'./models/{file_name}')

    def load_model(self, path):
        return joblib.load(path)

    def train_test_time_series(self, ts):
        """
        Split a time series into training and testing sets.

        Parameters
        ----------
        ts : pandas.Series
            The time series data to split.

        Returns
        -------
        tuple
            A tuple containing two pandas.Series objects, the training set
            and the testing set.
        """
        # Train Test Split Index
        train_size = 0.8
        split_idx = round(len(ts) * train_size)
        split_idx
        # Split
        train = ts.iloc[:split_idx]
        test = ts.iloc[split_idx:]
        return train, test

    def evaluate_forecast(self, actual, predicted, print_table=True, use_markdown=False):
        """
        Evalúa las métricas de error entre los valores reales y los valores predichos de una serie temporal.

        Args:
            actual (array-like): Valores reales de la serie temporal.
            predicted (array-like): Valores predichos de la serie temporal.
            print_table (bool, optional): Indica si se debe imprimir la tabla de métricas. Por defecto es True.
            use_markdown (bool, optional): Indica si se debe utilizar el formato Markdown en la impresión de la tabla.
                                        Por defecto es False.

        Returns:
            dict: Un diccionario que contiene las métricas de error calculadas.

        """
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        RMSE = mean_squared_error(actual, predicted, squared=False)
        MAPE = mean_absolute_percentage_error(actual, predicted)
        SMAPE = smape(actual, predicted)
        MAPE_percent = round(MAPE * 100, 2)

        # Configurar locale para mostrar los valores en pesos colombianos
        locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')

        # Crear una tabla con las métricas
        table_data = [
            ["**Scale-Dependent Metrics**", ""],
            ["Mean Squared Error (MSE):", locale.currency(mse, grouping=True)],
            ["Mean Absolute Error (MAE):", locale.currency(
                mae, grouping=True)],
            ["Root Mean Squared Error (RMSE):", locale.currency(
                RMSE, grouping=True)],
            ["**Percentage-Error Metrics**", ""],
            ["Mean Absolute percentage Error (MAPE)", f'{MAPE_percent}%'],
            ["Symmetric Mean Absolute percentage Error (SMAPE)",
             f'{round(SMAPE, 2)}%']
        ]

        if print_table:
            if use_markdown:
                markdown_table = tabulate(
                    table_data, headers=["Metric", "Value"], tablefmt="pipe")
                display(Markdown(markdown_table))
            else:
                print(tabulate(table_data, headers=[
                      "Metric", "Value"], tablefmt="fancy_grid"))

        # Retornar las métricas en un diccionario
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': RMSE,
            'MAPE': MAPE,
            'SMAPE': SMAPE,
            'MAPE_percent': MAPE_percent
        }

        return metrics

    def plot_time_series_model(self, train_y, test_y, y_pred, output_dir, names_folder, title='Serie de Tiempo - Train, Test y Predicciones'):
        """
        Grafica los datos de entrenamiento (train), prueba (test) y las predicciones de un modelo de series de tiempo.

        Args:
            train_y (pandas.Series): Serie de tiempo de los datos de entrenamiento.
            test_y (pandas.Series): Serie de tiempo de los datos de prueba.
            y_pred (pandas.Series): Serie de tiempo de las predicciones.

        Returns:
            None (Muestra el gráfico utilizando Plotly)

        """
        if not isinstance(train_y, pd.core.series.Series):
            raise TypeError("El objeto  train_y no es una instancia de pandas.core.series.Series.")
        if not isinstance(test_y, pd.core.series.Series):
            raise TypeError("El objeto  test_y no es una instancia de pandas.core.series.Series.")
        if not isinstance(y_pred, pd.core.series.Series):
            raise TypeError("El objeto  y_pred no es una instancia de pandas.core.series.Series.")
        # Concatenar los datos de entrenamiento, prueba y predicciones en un solo DataFrame
        df = pd.DataFrame(
            {'Train': train_y, 'Test': test_y, 'Predictions': y_pred})

        # Crear una figura de Plotly
        fig = go.Figure()

        # Agregar la serie de tiempo de entrenamiento al gráfico
        fig.add_trace(go.Scatter(x=train_y.index, y=train_y, name='Train'))

        # Agregar la serie de tiempo de prueba al gráfico
        fig.add_trace(go.Scatter(x=test_y.index, y=test_y, name='Test'))

        # Agregar la serie de tiempo de predicciones al gráfico
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, name='Predictions'))

        # Personalizar el diseño del gráfico
        fig.update_layout(title=title,
                          xaxis_title='Fecha',
                          yaxis_title='Valor',
                          legend=dict(x=0, y=1),
                          height=500)

    # Mostrar el gráfico
        country = names_folder['country']
        store = names_folder['store']
        model = names_folder['model_name']
        product = names_folder['product']

        # Si se proporciona un directorio de salida, guardar la figura en el archivo dentro de la carpeta correspondiente
        if output_dir:
            # Crear carpetas para país y tienda si no existen
            output_country_dir = os.path.join(output_dir, country)
            output_store_dir = os.path.join(output_country_dir, store)
            output_product_dir = os.path.join(output_store_dir, product)
            output_model_dir = os.path.join(output_product_dir, model)
            os.makedirs(output_model_dir, exist_ok=True)

            # Crear el nombre del archivo de salida en función del país y la tienda
            output_file = os.path.join(
                output_model_dir, f'{model}_model.html')
            fig.write_html(output_file)
        else:
            fig.show()

    def generate_folder_structure(self, folder_path: str, exclude_dirs: list = []) -> str:
        """
        Generate a text representation of the folder structure starting from the given folder path,
        excluding the directories specified in the exclude_dirs list.
        """
        def _generate_folder_structure_aux(folder_path, padding):
            folder_structure = ""
            folder_contents = sorted(os.listdir(folder_path))
            for i, item in enumerate(folder_contents):
                if item in exclude_dirs:
                    continue
                full_path = os.path.join(folder_path, item)
                if os.path.isdir(full_path):
                    if i == len(folder_contents) - 1:
                        folder_structure += f"{padding}└── {item}/\n"
                        folder_structure += _generate_folder_structure_aux(
                            full_path, padding + "    ")
                    else:
                        folder_structure += f"{padding}├── {item}/\n"
                        folder_structure += _generate_folder_structure_aux(
                            full_path, padding + "│   ")
                else:
                    if i == len(folder_contents) - 1:
                        folder_structure += f"{padding}└── {item}\n"
                    else:
                        folder_structure += f"{padding}├── {item}\n"
            return folder_structure

        folder_structure = _generate_folder_structure_aux(folder_path, "")
        file_path = os.path.join(folder_path, "folder_structure.txt")
        with open(file_path, "w") as f:
            f.write(folder_structure)
        return folder_structure

    def save_dataframe_as_csv(self, df, directory, file_name):
        """
        Save a DataFrame to a CSV file while preserving the index and date frequency.

        Args:
            df (pandas.DataFrame): The DataFrame to be saved.
            directory (str): The directory path where the CSV file will be saved.
            file_name (str): The name of the CSV file.

        Returns:
            None
        """

        if not os.path.isdir(directory):
            raise ValueError(f"No existe el directorio: '{directory}'.")
            # os.mkdir(directory)
        file_path_to_save = os.path.join(directory, file_name)
        df.to_csv(file_path_to_save, index=True, date_format='%Y-%m-%d')

    def save_results_to_file(self, model_metrics, directory, file_name):
        if not os.path.isdir(directory):
            raise ValueError(f"No existe el directorio: '{directory}'.")
        result_df = pd.DataFrame(model_metrics)
        file_path_to_save = os.path.join(directory, file_name)
        result_df.to_csv(file_path_to_save, index=False)

    def plot_time_series_individual_model(self, train_y, test_y, y_pred, output_dir, names_folder, title='Serie de Tiempo - Train, Test y Predicciones'):
        """
        Grafica los datos de entrenamiento (train), prueba (test) y las predicciones de un modelo de series de tiempo.

        Args:
            train_y (pandas.Series): Serie de tiempo de los datos de entrenamiento.
            test_y (pandas.Series): Serie de tiempo de los datos de prueba.
            y_pred (pandas.Series): Serie de tiempo de las predicciones.

        Returns:
            None (Muestra el gráfico utilizando Plotly)

        """
        if not isinstance(train_y, pd.core.series.Series):
            raise TypeError("El objeto  train_y no es una instancia de pandas.core.series.Series.")
        if not isinstance(test_y, pd.core.series.Series):
            raise TypeError("El objeto  test_y no es una instancia de pandas.core.series.Series.")
        if not isinstance(y_pred, pd.core.series.Series):
            raise TypeError("El objeto  y_pred no es una instancia de pandas.core.series.Series.")
        # Concatenar los datos de entrenamiento, prueba y predicciones en un solo DataFrame
        df = pd.DataFrame(
            {'Train': train_y, 'Test': test_y, 'Predictions': y_pred})

        # Crear una figura de Plotly
        fig = go.Figure()

        # Agregar la serie de tiempo de entrenamiento al gráfico
        fig.add_trace(go.Scatter(x=train_y.index, y=train_y, name='Train'))

        # Agregar la serie de tiempo de prueba al gráfico
        fig.add_trace(go.Scatter(x=test_y.index, y=test_y, name='Test'))

        # Agregar la serie de tiempo de predicciones al gráfico
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, name='Predictions'))

        # Personalizar el diseño del gráfico
        fig.update_layout(title=title,
                          xaxis_title='Fecha',
                          yaxis_title='Valor',
                          legend=dict(x=0, y=1),
                          height=500)

    # Mostrar el gráfico

        model = names_folder['model_name']


        # Si se proporciona un directorio de salida, guardar la figura en el archivo dentro de la carpeta correspondiente
        if output_dir:
            # Crear carpetas para país y tienda si no existen
            output_model_dir = os.path.join(output_dir, model)
            os.makedirs(output_model_dir, exist_ok=True)

            # Crear el nombre del archivo de salida en función del país y la tienda
            output_file = os.path.join(
                output_model_dir, f'{model}_model.html')
            fig.write_html(output_file)
        else:
            fig.show()

