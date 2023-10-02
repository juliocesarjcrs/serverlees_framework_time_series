import pandas as pd
from src.core.dependency_inyections.container import Container
from src.utils.utils import Utils
from src.enums.models_type import ModelsType
from src.utils.logger.logger import Logger


class DataSelectionFacade:

    def __init__(self, data: pd.DataFrame):
        # Crea una instancia del contenedor y resuelve las dependencias
        self.container = Container()
        self.utils = Utils()
        self.model_training = self.container.model_training()
        self.time_series_feature_engineering = self.container.time_series_feature_engineering()
        self.time_series_outlier_detector = self.container.time_series_outlier_detector()
        self.dataframe = data
        self.logger = Logger("DataSelectionFacade")

    def train_models_individual(self, date_col_name, target_col: str, freq, train_models: list) -> list:

        processed_df = self.add_complete_features_enginiering()
        # Set models
        for model_type in train_models:
            if model_type == ModelsType.XGBOOST:
                self.model_training.set_model_xgboost()
                self.logger.info(f'::: SET MODEL::: {ModelsType.XGBOOST.value}')
            elif model_type == ModelsType.LINEAR_REGRESSION:
                self.model_training.set_model_linear_regression()
                self.logger.info(f'::: SET MODEL::: {ModelsType.LINEAR_REGRESSION.value}')
            elif model_type == ModelsType.BOOSTED_HIBRID:
                self.model_training.set_model_boosted_hybrid()
                self.logger.info(f'::: SET MODEL::: {ModelsType.BOOSTED_HIBRID.value}')


        # train an evaluate
        self.model_training.set_individual_dataset(processed_df, target_col)
        self.model_training.train_and_evaluate(target_col)
        # save metrics
        model_metrics = self.model_training.get_model_metrics()
        return model_metrics

    def add_complete_features_enginiering(self):
        processed_df, _ = self.time_series_feature_engineering.feature_engineering_time_series_dynamic(
            self.dataframe)
        return self.time_series_feature_engineering.add_monthly_holiday_count(processed_df, 'holidays_cont')

    def get_best_model(self, target_col: str, best_model_name: str) -> dict:

        processed_df = self.add_complete_features_enginiering()
        # Set models
        if ModelsType.XGBOOST.value == best_model_name:
            self.model_training.set_model_xgboost()
        elif ModelsType.LINEAR_REGRESSION.value == best_model_name:
            self.model_training.set_model_linear_regression()
        else:
            raise ValueError(
                f'best model name does not definided ${best_model_name}')

        # train an evaluate
        self.model_training.set_individual_dataset(processed_df, target_col)
        result = self.model_training.train_and_evaluate(target_col, True)
        return result

