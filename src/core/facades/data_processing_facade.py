import pandas as pd
from src.core.dependency_inyections.container import Container
from src.utils.utils import Utils
from src.enums.models_type import ModelsType


class DataProcessingFacade:

    def __init__(self, data: pd.DataFrame):
        # Crea una instancia del contenedor y resuelve las dependencias
        self.container = Container()
        self.utils = Utils()
        self.model_training = self.container.model_training()
        self.time_series_feature_engineering = self.container.time_series_feature_engineering()
        self.dataframe = data

    def train_models_individual(self, date_col_name, target_col: str, freq) -> list:

        processed_df, _ = self.time_series_feature_engineering.feature_engineering_time_series_dynamic(
            self.dataframe)
        # Set models
        self.model_training.set_model_xgboost()
        self.model_training.set_model_linear_regression()

        # train an evaluate
        self.model_training.set_individual_dataset(processed_df, target_col)
        self.model_training.train_and_evaluate(target_col)
        # save metrics
        model_metrics = self.model_training.get_model_metrics()
        return model_metrics

    def get_best_model(self, target_col: str, best_model_name: str) -> dict:

        processed_df, _ = self.time_series_feature_engineering.feature_engineering_time_series_dynamic(
            self.dataframe)
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
