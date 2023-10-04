import pandas as pd
from src.core.dependency_inyections.container import Container
from src.utils.utils import Utils
from src.utils.logger.logger import Logger
from src.enums.anomaly_detection_method import AnomalyDetectionMethod


class DataProcessingFacade:

    def __init__(self, data: pd.DataFrame):
        # Crea una instancia del contenedor y resuelve las dependencias
        self.container = Container()
        self.utils = Utils()
        self.model_training = self.container.model_training()
        self.time_series_feature_engineering = self.container.time_series_feature_engineering()
        self.time_series_outlier_detector = self.container.time_series_outlier_detector()
        self.dataframe = data
        self.logger = Logger("DataProcessingFacade")

    def get_data_without_outliers(self, target_col: str, anomaly_detection_method) -> pd.DataFrame:
        if AnomalyDetectionMethod.DBSCAN.value == anomaly_detection_method:
            anomalies = self.time_series_outlier_detector.detect_anomalies_by_DBSSCAN(
                self.dataframe, target_col)
        elif AnomalyDetectionMethod.Z_SCORE.value == anomaly_detection_method:
            anomalies = self.time_series_outlier_detector.detect_anomalies(
                self.dataframe, target_col)
        else:
            raise ValueError(f'best model name does not definided ${anomaly_detection_method}')

        imputed_series = self.time_series_outlier_detector.time_series_interpolation(
            self.dataframe, anomalies.index)
        df_time_without_outlier = self.dataframe
        df_time_without_outlier[target_col] = imputed_series
        return df_time_without_outlier, anomalies
