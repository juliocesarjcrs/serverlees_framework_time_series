import pandas as pd
from sklearn.base import BaseEstimator
# Models
import statsmodels.tsa.api as tsa
import pmdarima as pm
from src.core.training.models.boosted_hybrid_model import BoostedHybridModel
# Error
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from src.error.exception.model_training_error import ModelTrainingError
from statsmodels.tsa.deterministic import DeterministicProcess
from src.utils.logger.logger import Logger

# Enums
from src.enums.models_type import ModelsType


class ModelTraining:

    def __init__(self, utils):
        self.utils = utils
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
        self.models.append(('Arima', tsa.ARIMA))
        self.order_arima = model_order

    def set_params_model_sarima(self, model_order_sarima):
        self.models.append(('SARIMA', tsa.SARIMAX))
        self.order_arima = model_order_sarima

    def set_model_auto_arima(self):
        self.models.append(('SARIMA-AutoArima', pm.auto_arima))

    def set_model_xgboost(self):
        self.models.append(
            ('XGBoost', xgb.XGBRegressor(early_stopping_rounds=50)))

    def set_model_linear_regression(self):
        self.models.append(
            (ModelsType.LINEAR_REGRESSION.value, LinearRegression()))

    def set_model_boosted_hybrid(self):
        model_hybrid = BoostedHybridModel(
            model_1=LinearRegression(),
            model_2=xgb.XGBRegressor()
        )
        self.models.append(
            ('BoostedHybrid', model_hybrid))

    def get_model_metrics(self):
        return self.model_metrics

    def get_names_folder(self):
        return self.names_folder

    def set_individual_dataset(self, dataframe: pd.DataFrame, target_col: str):
        X = dataframe.drop(target_col, axis=1)
        y = dataframe[target_col]
        self.datasets = [
            {'name': 'Normal', 'train_ratio': 0.8, 'X': X, 'y': y},
            # {'name': 'Without Outliers', 'train_ratio': 0.8, 'X': X_without_outliers, 'y': y_without_outliers}
            # Add more datasets if needed
        ]

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

                result, y_pred = self.evaluate_model(
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
            model_hybrid_fit = model.fit(x_train_deterministic, train_x, y_train_deterministic)
            return model_hybrid_fit, parameters
        else:
            raise ValueError(f'Undefined type of model: ${model_name}')

    def predict_model(self, model: BaseEstimator, test_x, model_name: str, target_col: str) -> pd.Series:
        """
        Realiza predicciones utilizando un modelo entrenado.

        Parameters:
            model (BaseEstimator): El modelo de machine learning entrenado.
            test_x (Any): Las características de prueba para las cuales se realizarán las predicciones.

        Returns:
            Any: Las predicciones realizadas por el modelo.
        """
        if model_name == ModelsType.LINEAR_REGRESSION.value:
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
        else:
            raise ValueError(f'Undefined type of model: ${model_name}')

    def evaluate_model(self, model_name, model, train_x, train_y, test_x, test_y, target_col, is_individual=True):
        try:
            self.logger.info(f"::: START FIT MODEL {model_name} :::")
            fitted_model, parameters = self.fit_model(
                model, model_name, train_x, train_y, test_x, test_y)
            self.logger.info(f"::: END FIT MODEL {model_name}   :::")

            self.logger.info(f"::: START PREDICT {model_name}   :::")
            y_pred = self.predict_model(
                fitted_model, test_x, model_name, target_col)
            self.logger.info(f"::: END PREDICT {model_name}     :::")

            if not isinstance(y_pred, pd.Series) or not isinstance(test_y, pd.Series):
                raise ValueError(
                    f"y_pred and test_y must be pandas Series. y_pred is of type {type(y_pred)}. test_y is of type {type(test_y)}")

            if y_pred.shape[0] != test_y.shape[0]:
                raise ValueError('Shapes of y_pred and test_y do not match.')

            metrics = self.utils.evaluate_forecast(test_y, y_pred, True)

            names_folder = {
                'model_name': model_name
            }

            if not is_individual:
                names_folder.update({
                    'country': self.country,
                    'store': self.store,
                    'product': self.product
                })
            should_save_graph = False
            if should_save_graph:
                self.utils.plot_time_series(
                    train_y, test_y, y_pred, '/code/reports/graficas_modelos', names_folder, f'Serie Tiempo - {model_name}')

            return {
                'model_name': model_name,
                'model_fit': fitted_model,
                'metrics': metrics,
                'parameters_training': parameters,
                'start_date_test': test_x.index[0].strftime('%Y-%m-%d')
            }, y_pred

        except Exception as exception:
            raise ModelTrainingError(
                model_name, f"Error al entrenar o evaluar el modelo {model_name}: {exception}")

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

    # def fit_auto_arima_model(self, model, train_y, order_arima):
    #     d = order_arima[1]
    #     auto_sarima_model = model(train_y,
    #                               start_p=1, start_q=1,
    #                               test='adf',
    #                               max_p=4, max_q=4, m=7,
    #                               start_P=0, seasonal=True,
    #                               d=d, D=1,
    #                               trace=False,
    #                               error_action='ignore',
    #                               suppress_warnings=True,
    #                               stepwise=True)
    #     return auto_sarima_model
