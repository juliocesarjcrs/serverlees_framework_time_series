import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from src.utils.logger.logger import Logger


class TimeSeriesOutlierDetector:
    def __init__(self):
        self.logger = Logger("TimeSeriesOutlierDetector")

    def time_series_interpolation(self, time_series, anomaly_index):
        """
        Performs interpolation on a time series by imputing missing values at the specified anomaly index.

        Parameters:
            time_series (pd.Series): The original time series.
            anomaly_index: The index of the anomaly where the missing value occurs.

        Returns:
            pd.Series: The time series with missing values imputed using linear interpolation.
        """
        # Create a copy of the original time series
        interpolated_series = time_series.copy()

        # Get the anomalous value
        anomaly_value = interpolated_series.loc[anomaly_index]

        # Replace the anomalous value with NaN (missing values)
        interpolated_series.loc[anomaly_index] = np.nan

        # Perform linear interpolation to impute the missing values
        interpolated_series.interpolate(method='linear', inplace=True)

        return interpolated_series

    def detect_anomalies(self, data: pd.DataFrame, target_col:str, threshold_factor=2)-> pd.DataFrame:
        """
        Detects anomalies in a specified column of a DataFrame using mean and standard deviation approach.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            target_col (str): The column name where anomalies should be detected.
            threshold_factor (float, optional): The factor to multiply the standard deviation to determine the threshold.

        Returns:
            pd.DataFrame: A DataFrame containing the rows with detected anomalies.
        """
        mean = data[target_col].mean()
        std = data[target_col].std()
        threshold = threshold_factor * std
        anomalies = data[data[target_col].abs() - mean > threshold]
        self.logger.info(f"::: Anomalies ::: {anomalies}")

        return anomalies

    def detect_anomalies_by_DBSSCAN(self, data: pd.DataFrame, target_col: str, eps=0.1)-> pd.DataFrame:
        x_data = data[target_col].values.reshape((-1, 1))
        scaler = StandardScaler()
        x_data_scaled = scaler.fit_transform(x_data)
        # Entrenar el modelo DBSCAN para la detección de anomalías
        clustering = DBSCAN(eps=eps, min_samples=2).fit(x_data_scaled)

        # Etiquetar los puntos como anomalías (etiqueta -1) o no (etiqueta 0, 1, 2, ...)
        labels = clustering.labels_

        # Identificar los índices de los puntos anómalos
        anomaly = data[labels == -1]
        self.logger.info(f"::: Anomalies ::: {anomaly}")
        return anomaly
