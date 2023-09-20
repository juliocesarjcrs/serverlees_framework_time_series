import os
import matplotlib.pyplot as plt
import pandas as pd


class DataExplorer:

    def __init__(self, utils):
        self.utils = utils
        # data_path = 'data/raw/train.csv'
        data_path = 'data/processed/df_time_monthly_without_outliers.csv'
        # self.data_path = data_path
        self.data = self.utils.load_from_csv(data_path, 'date')

    def get_dataframe(self):
        return self.data

    def plot_time_series(self, group_cols, target_col, output_file=None):
        if target_col not in self.data.columns:
            raise ValueError(
                f"La columna objetivo '{target_col}' no existe en el DataFrame.")

        # Obtener la lista de países únicos en los datos
        unique_countries = self.data['country'].unique()

        for country in unique_countries:
            # Filtrar los datos solo para el país actual
            country_data = self.data[self.data['country'] == country]

            # Crear una figura y ejes separados para cada país
            fig, axes = plt.subplots(figsize=(14, 8))

            # Iterar sobre cada grupo único y trazar la serie temporal correspondiente
            for group, group_data in country_data.groupby(group_cols):
                if not isinstance(group_data.index, pd.DatetimeIndex):
                    raise ValueError(
                        "El índice del DataFrame no es de tipo DatetimeIndex. Asegúrate de que tus fechas estén en el índice.")

                # Ordenar por fecha para asegurarse de que el gráfico esté ordenado cronológicamente
                group_data = group_data.sort_index()

                # Plotear la serie temporal de ventas para este grupo
                axes.plot(group_data.index, group_data[target_col],
                          label=f"{', '.join(str(val) for val in group)}")

            # Establecer títulos y etiquetas del eje
            axes.set_title(
                f'Ventas diarias en {country} por ' + ', '.join(group_cols))
            axes.set_xlabel('Fecha')
            axes.set_ylabel(f'Cantidad de {target_col}')

            # Crear leyenda fuera del gráfico principal
            axes.legend(loc='upper left', bbox_to_anchor=(
                1, 1), fontsize='small')

            # Rotar las fechas en el eje x para una mejor visualización
            plt.xticks(rotation=45)

            # Mostrar el gráfico o guardar la figura en el archivo
            if output_file:
                output_file_country = output_file.replace(
                    '.png', f'_{country}.png')
                plt.savefig(output_file_country)
                plt.close()  # Cerrar la figura para liberar memoria
            else:
                plt.tight_layout()
                plt.show()

    def plot_time_series_by_country_store(self, group_cols, target_col, output_dir=None):
        if target_col not in self.data.columns:
            raise ValueError(
                f"La columna objetivo '{target_col}' no existe en el DataFrame.")

        # Obtener la lista de países únicos en los datos
        unique_countries = self.data['country'].unique()

        for country in unique_countries:
            # Filtrar los datos solo para el país actual
            country_data = self.data[self.data['country'] == country]

            # Obtener la lista de tiendas únicas en los datos para este país
            unique_stores = country_data['store'].unique()

            for store in unique_stores:
                # Filtrar los datos solo para la tienda actual
                store_data = country_data[country_data['store'] == store]

                # Crear una figura y ejes separados para cada tienda
                fig, axes = plt.subplots(figsize=(14, 8))

                # Iterar sobre cada grupo único y trazar la serie temporal correspondiente
                for group, group_data in store_data.groupby(group_cols):
                    if not isinstance(group_data.index, pd.DatetimeIndex):
                        raise ValueError(
                            "El índice del DataFrame no es de tipo DatetimeIndex. Asegúrate de que tus fechas estén en el índice.")

                    # Ordenar por fecha para asegurarse de que el gráfico esté ordenado cronológicamente
                    group_data = group_data.sort_index()

                    # Plotear la serie temporal de ventas para este grupo
                    axes.plot(
                        group_data.index, group_data[target_col], label=f"{', '.join(str(val) for val in group)}")

                # Establecer títulos y etiquetas del eje
                axes.set_title(
                    f'Ventas diarias en {country}, tienda {store} por ' + ', '.join(group_cols))
                axes.set_xlabel('Fecha')
                axes.set_ylabel(f'Cantidad de {target_col}')

                # Crear leyenda fuera del gráfico principal
                axes.legend(loc='upper left', bbox_to_anchor=(
                    1, 1), fontsize='small')

                # Rotar las fechas en el eje x para una mejor visualización
                plt.xticks(rotation=45)

                # Si se proporciona un directorio de salida, guardar la figura en el archivo dentro de la carpeta correspondiente
                if output_dir:
                    # Crear carpetas para país y tienda si no existen
                    output_country_dir = os.path.join(output_dir, country)
                    output_store_dir = os.path.join(output_country_dir, store)
                    os.makedirs(output_store_dir, exist_ok=True)

                    # Crear el nombre del archivo de salida en función del país y la tienda
                    output_file = os.path.join(
                        output_store_dir, f'{target_col}_ventas.png')

                    plt.savefig(output_file)
                    plt.close()  # Cerrar la figura para liberar memoria
                else:
                    plt.tight_layout()
                    plt.show()
