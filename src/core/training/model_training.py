import pandas as pd
from sklearn.base import BaseEstimator
# Models
import statsmodels.tsa.api as tsa
import pmdarima as pm
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
from src.core.training.models.boosted_hybrid_model import BoostedHybridModel
from sklearn.ensemble import RandomForestRegressor
# Error
from src.utils.logger.logger import Logger

# Enums
from src.enums.models_type import ModelsType

# types
from src.types.training.model_training_types import NamesFolder, OptionsSavePlot
from typing import Dict, List
# fastAPi
from fastapi import HTTPException


class ModelTraining:

    def __init__(self, utils, type_storage):
        self.utils = utils
        self.type_storage = type_storage
        self.datasets = []
        self.model_metrics = []
        self.models = []
        self.order_arima = None
        self.country = None
        self.store = None
        self.product = None
        self.names_folder = None
        self.target = None
        self.final_submission = []
        self.x_test_deterministic = None
        self.logger = Logger("ModelTraining")

    def set_params_model_arima(self, model_order):
        self.models.append((ModelsType.ARIMA.value, tsa.ARIMA))
        self.order_arima = model_order

    def set_params_model_sarima(self, model_order_sarima):
        self.models.append((ModelsType.SARIMA.value, tsa.SARIMAX))
        self.order_arima = model_order_sarima

    def set_model_auto_arima(self):
        self.models.append((ModelsType.SARIMA_AUTO_ARIMA.value, pm.auto_arima))

    def set_model_xgboost(self):
        self.models.append(
            (ModelsType.XGBOOST.value, xgb.XGBRegressor(early_stopping_rounds=50)))

    def set_model_linear_regression(self):
        self.models.append(
            (ModelsType.LINEAR_REGRESSION.value, LinearRegression()))

    def set_model_boosted_hybrid(self):
        model_hybrid = BoostedHybridModel(
            model_1=LinearRegression(),
            model_2=xgb.XGBRegressor()
        )
        self.models.append(
            (ModelsType.BOOSTED_HIBRID.value, model_hybrid))

    def set_model_Random_forest_regressor(self):
        self.models.append(
            (ModelsType.RANDOM_FOREST_REGRESSOR.value, RandomForestRegressor()))

    def get_model_metrics(self):
        return self.model_metrics

    def get_names_folder(self):
        return self.names_folder

    def set_dataset(self, params):
        self.datasets.append(params)

    def reset_dataset(self):
        self.datasets = []

    def reset_models(self):
        self.models = []

    def train_and_evaluate(self, target_col, with_model: bool = False):
        for dataset in self.datasets:
            dataset_name = dataset['name']
            for model_name, model_class in self.models:
                model = model_class
                train_ratio = dataset['train_ratio']
                train_x, test_x = self.split_train_test_data(
                    dataset['X'], train_ratio)
                train_y, test_y = self.split_train_test_data(
                    dataset['y'], train_ratio)

                result, y_pred = self.evaluate_model(dataset_name,
                                                     model_name, model, train_x, train_y, test_x, test_y, target_col)
                if with_model:
                    return result
                else:
                    result_without_model_fit = {
                        key: value for key, value in result.items() if key != 'model_fit'}
                    self.model_metrics.append(result_without_model_fit)

    def split_train_test_data(self, data, train_ratio):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    def fit_model(self, model: BaseEstimator, model_name: str, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series) -> BaseEstimator:
        """
        Entrena un modelo de machine learning con datos de entrenamiento.

        Parameters:
            model (BaseEstimator): El modelo de machine learning que se va a entrenar.
            train_x (Any): Las características de entrenamiento.
            train_y (Any): Las etiquetas de entrenamiento.

        Returns:
            BaseEstimator: El modelo entrenado.
        """
        parameters = {
            'train_columns': train_x.columns.tolist()
        }
        if model_name == ModelsType.XGBOOST.value:
            return model.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], verbose=False), parameters
        elif model_name == ModelsType.LINEAR_REGRESSION.value:
            return model.fit(train_x, train_y), parameters
        elif model_name == ModelsType.RANDOM_FOREST_REGRESSOR.value:
            return model.fit(train_x, train_y), parameters
        elif model_name == ModelsType.BOOSTED_HIBRID.value:
            y_combined = pd.concat([train_y, test_y])

            deterministic_process = DeterministicProcess(index=y_combined.index,  # dates from the training data
                                                         constant=True,  # the intercept
                                                         order=3,        # quadratic trend
                                                         drop=True)
            deterministic_values = deterministic_process.in_sample()
            x_train_deterministic, x_test_deterministic = self.split_train_test_data(
                deterministic_values, 0.8)
            self.x_test_deterministic = x_test_deterministic

            y_train_deterministic, y_test_deterministic = self.split_train_test_data(
                y_combined, 0.8)
            model_hybrid_fit = model.fit(
                x_train_deterministic, train_x, y_train_deterministic)
            return model_hybrid_fit, parameters
        elif model_name == ModelsType.SARIMA_AUTO_ARIMA.value:
            return self.fit_auto_arima_model(model, train_y), parameters
        else:
            self.logger.error(
                f'model_name is not defined by fit: {model_name}')
            raise HTTPException(
                status_code=400,
                detail=f'model_name is not defined by fit: {model_name}'
            )

    def predict_model(self, model: BaseEstimator, data: Dict[str, List[float]], model_name: str, target_col: str) -> pd.Series:
        """
        Args:
            fitted_model (Any): El modelo previamente ajustado.
            data (Dict[str, List[float]]): Un diccionario que contiene datos de prueba.
            model_name (str): El nombre del modelo.
            target_col (str): El nombre de la columna objetivo.

        Returns:
            Any: Las predicciones realizadas por el modelo.
        """
        test_x = data['test_x']
        test_y = data['test_y']
        if model_name == ModelsType.LINEAR_REGRESSION.value:
            y_pred_values = model.predict(test_x)
            return pd.Series(y_pred_values, index=test_x.index)
        elif model_name == ModelsType.RANDOM_FOREST_REGRESSOR.value:
            y_pred_values = model.predict(test_x)
            return pd.Series(y_pred_values, index=test_x.index)
        elif model_name == ModelsType.XGBOOST.value:
            x_test = test_x
            y_pred_values = model.predict(x_test)
            y_pred = pd.DataFrame(
                {target_col: y_pred_values.astype('int')}, index=x_test.index)
            return y_pred[target_col]
        elif model_name == ModelsType.BOOSTED_HIBRID.value:
            return model.predict(self.x_test_deterministic, test_x)
        elif model_name == ModelsType.SARIMA_AUTO_ARIMA.value:
            return model.predict(n_periods=test_y.shape[0])
        else:
            self.logger.error(
                f'model_name is not defined by predict: {model_name}')
            # raise ValueError(f'Undefined type of model: ${model_name}')
            raise HTTPException(
                status_code=400,
                detail=f'model_name is not defined by predict: {model_name}'
            )

    def evaluate_model(self, dataset_name: str, model_name: str, model, train_x, train_y, test_x, test_y, target_col, is_individual=True):
        try:
            self.logger.info(f"::: START FIT MODEL {model_name} :::")
            fitted_model, parameters = self.fit_model(
                model, model_name, train_x, train_y, test_x, test_y)
            self.logger.info(f"::: END FIT MODEL   {model_name}   :::")

            self.logger.info(f"::: START PREDICT {model_name}   :::")
            data = {
                'test_x': test_x,
                'test_y': test_y
            }
            y_pred = self.predict_model(
                fitted_model, data, model_name, target_col)
            self.logger.info(f"::: END PREDICT   {model_name}     :::")

            if not isinstance(y_pred, pd.Series) or not isinstance(test_y, pd.Series):
                raise ValueError(
                    f"y_pred and test_y must be pandas Series. y_pred is of type {type(y_pred)}. test_y is of type {type(test_y)}")

            if y_pred.shape[0] != test_y.shape[0]:
                raise ValueError('Shapes of y_pred and test_y do not match.')

            metrics = self.utils.evaluate_forecast(test_y, y_pred, True)

            names_folder: NamesFolder = {
                'model_name': model_name,
                'dataset_name': dataset_name
            }

            if not is_individual:
                names_folder.update({
                    'country': self.country,
                    'store': self.store,
                    'product': self.product
                })
            should_save_graph = True
            if should_save_graph:
                options_save_plot: OptionsSavePlot = {
                    'type_storage': self.type_storage,
                    'output_dir': 'reports/models-graph',
                    'names_folder': names_folder,
                    'title':  f'Serie Tiempo  -{dataset_name}-{model_name}'
                }
                self.utils.plot_time_series_individual_model(
                    train_y, test_y, y_pred, options_save_plot)

            return {
                'dataset_name': dataset_name,
                'model_name': model_name,
                'model_fit': fitted_model,
                'metrics': metrics,
                'parameters_training': parameters,
                'start_date_test': test_x.index[0].strftime('%Y-%m-%d')
            }, y_pred

        except Exception as exception:
            raise HTTPException(
                status_code=400,
                detail=f'Error al entrenar o evaluar el modelo {model_name}: {exception}'
            )
            # raise ModelTrainingError(
            #     model_name, f"Error al entrenar o evaluar el modelo {model_name}: {exception}")

    # def evaluate_model(self, model_name: str, model, train_X, train_y, test_X, test_y, target_col, is_individual=True):
    #     try:
    #
    #         if model_name == "Arima":
    #             fitted_model = self.fit_arima_model(
    #                 model, train_y, self.order_arima)
    #         elif model_name == "SARIMA":
    #             fitted_model = self.fit_sarima_model(
    #                 model, train_y, self.order_arima)
    #         elif model_name == "SARIMA-AutoArima":
    #             fitted_model = self.fit_auto_arima_model(
    #                 model, train_y, self.order_arima)
    #         elif model_name == "XGBoost":
    #             fitted_model = self.fit_xgboost_model(
    #                 model, train_X, train_y, test_X, test_y)
    #         elif model_name == ModelsType.LINEAR_REGRESSION.value:
    #             fitted_model = self.fit_linear_regression_model(
    #                 model, train_X, train_y)

    #

    #         y_pred = None
    #
    #         if model_name == "Arima" or model_name == "SARIMA":
    #             y_pred = fitted_model.predict(
    #                 start=len(train_y), end=len(train_y) + len(test_y) - 1)

    #         elif model_name == "SARIMA-AutoArima":
    #             y_pred = fitted_model.predict(n_periods=test_y.shape[0])

    #         elif model_name == "XGBoost":
    #
    #             y_pred_values = fitted_model.predict(X_test)
    #             y_pred = pd.DataFrame(
    #                 {target_col: y_pred_values.astype('int')}, index=X_test.index)
    #             y_pred = y_pred[target_col]

    #         elif model_name == ModelsType.LINEAR_REGRESSION.value:
    #             y_pred_values = fitted_model.predict(test_X)
    #             y_pred = pd.Series(y_pred_values, index=test_X.index)

    #
    #         if not isinstance(y_pred, pd.Series):
    #             raise ValueError("y_pred no es una serie")

    #         if not isinstance(test_y, pd.Series):
    #             raise ValueError("test_y no es una serie")

    #         if y_pred.shape[0] != test_y.shape[0]:
    #             raise ValueError(
    #                 f"La forma de y_pred:'{y_pred.shape[0]}' No coincide con test_y:'{test_y.shape[0]}'.")

    #         metrics = self.utils.evaluate_forecast(test_y, y_pred, True)
    #         if is_individual:
    #             names_folder = {
    #                 'model_name': model_name
    #             }
    #             self.utils.plot_time_series_individual_model(
    #                 train_y, test_y, y_pred, '/code/reports/graficas_modelos', names_folder, f'Serie Tiempo  -{model_name}')

    #         else:
    #             names_folder = {
    #                 'country': self.country,
    #                 'store': self.store,
    #                 'product': self.product,
    #                 'model_name': model_name
    #             }
    #             self.utils.plot_time_series_model(
    #                 train_y, test_y, y_pred, '/code/reports/graficas_modelos', names_folder, f'Serie Tiempo  -{model_name}')

    #         return {
    #             'model_name': model_name,
    #             'model_fit': model,
    #             'metrics': metrics
    #         }, y_pred
    #     except Exception as e:
    #
    #         return None, None

    # def fit_arima_model(self, model, train_y, order_arima):
    #     return model(train_y, order=order_arima).fit()

    # def fit_sarima_model(self, model, train_y, order_arima):
    #     return model(train_y, order=order_arima, seasonal_order=(P, D, Q, m), trend='c')

    def fit_auto_arima_model(self, model, train_y):

        auto_sarima_model = model(train_y,
                                  start_p=1, start_q=1,
                                  test='adf',
                                  max_p=4, max_q=4, m=7,
                                  start_P=0, seasonal=True,
                                  d=1, D=1,
                                  trace=False,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)
        return auto_sarima_model
