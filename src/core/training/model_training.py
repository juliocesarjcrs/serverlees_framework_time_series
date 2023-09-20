import pandas as pd
from sklearn.base import BaseEstimator
# Models
import statsmodels.tsa.api as tsa
import pmdarima as pm
# Error
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from src.error.exception.model_training_error import ModelTrainingError


# Enums
from src.enums.models_type import ModelsType


class ModelTraining:

    def __init__(self, utils):
        self.utils = utils
        self.data = None
        self.datasets = []
        self.model_metrics = []
        self.models = []
        self.order_arima = None
        self.country = None
        self.store = None
        self.product = None
        self.names_folder = None
        self.features = None
        self.target = None
        self.final_submission = []

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
        self.features = ['day_of_week', 'month', 'day', 'year']
        self.target = ['num_sold']

    def set_model_linear_regression(self):
        self.models.append(
            (ModelsType.LINEAR_REGRESSION.value, LinearRegression()))

    def get_model_metrics(self):
        return self.model_metrics

    def get_names_folder(self):
        return self.names_folder


    def set_individual_dataset(self, data, target_col):
        X = data.drop(target_col, axis=1)
        y = data[target_col]
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
                    result_without_model_fit = {key: value for key, value in result.items() if key != 'model_fit'}
                    self.model_metrics.append(result_without_model_fit)

    def fit_model(self, model: BaseEstimator, model_name: str, train_x, train_y, test_x, test_y) -> BaseEstimator:
        """
        Entrena un modelo de machine learning con datos de entrenamiento.

        Parameters:
            model (BaseEstimator): El modelo de machine learning que se va a entrenar.
            train_x (Any): Las características de entrenamiento.
            train_y (Any): Las etiquetas de entrenamiento.

        Returns:
            BaseEstimator: El modelo entrenado.
        """
        if model_name == ModelsType.XGBOOST.value:
            return model.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], verbose=False)
        else:
            return model.fit(train_x, train_y)

    def predict_model(self, model: BaseEstimator, test_x, model_name:str, target_col: str)-> pd.Series:
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
            # x_test = test_x[self.features]
            x_test = test_x
            y_pred_values = model.predict(x_test)
            y_pred = pd.DataFrame(
                {target_col: y_pred_values.astype('int')}, index=x_test.index)
            return y_pred[target_col]
        else:
            return model.predict(test_x)

    def evaluate_model(self, model_name, model, train_x, train_y, test_x, test_y, target_col, is_individual=True):
        try:
            print(f"::: START FIT MODEL {model_name} :::")
            fitted_model = self.fit_model(model, model_name, train_x, train_y, test_x, test_y)
            print(f"::: END FIT MODEL {model_name}   :::")

            print(f"::: START PREDICT {model_name}   :::")
            y_pred = self.predict_model(fitted_model, test_x, model_name, target_col)
            print(f"::: END PREDICT {model_name}     :::")

            if not isinstance(y_pred, pd.Series) or not isinstance(test_y, pd.Series):
                raise ValueError(f"y_pred and test_y must be pandas Series. y_pred is of type {type(y_pred)}. test_y is of type {type(test_y)}")

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
                'metrics': metrics
            }, y_pred

        except Exception as exception:
            raise ModelTrainingError(model_name, f"Error al entrenar o evaluar el modelo {model_name}: {exception}")
            # print(f"Error al entrenar o evaluar el modelo {model_name}: {exception}")
            # return None, None

    # def evaluate_model(self, model_name: str, model, train_X, train_y, test_X, test_y, target_col, is_individual=True):
    #     try:
    #         print(":: START FIT MODEL " + model_name)
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

    #         print(":: END FIT MODEL " + model_name)

    #         y_pred = None
    #         print(":: START PREDICT " + model_name)
    #         if model_name == "Arima" or model_name == "SARIMA":
    #             y_pred = fitted_model.predict(
    #                 start=len(train_y), end=len(train_y) + len(test_y) - 1)

    #         elif model_name == "SARIMA-AutoArima":
    #             y_pred = fitted_model.predict(n_periods=test_y.shape[0])

    #         elif model_name == "XGBoost":
    #             X_test = test_X[self.features]
    #             y_pred_values = fitted_model.predict(X_test)
    #             y_pred = pd.DataFrame(
    #                 {target_col: y_pred_values.astype('int')}, index=X_test.index)
    #             y_pred = y_pred[target_col]

    #         elif model_name == ModelsType.LINEAR_REGRESSION.value:
    #             y_pred_values = fitted_model.predict(test_X)
    #             y_pred = pd.Series(y_pred_values, index=test_X.index)

    #         print(":: END PREDICT " + model_name)
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
    #         print(f"Error al entrenar o evaluar el modelo {model_name}: {e}")
    #         return None, None

    def fit_arima_model(self, model, train_y, order_arima):
        return model(train_y, order=order_arima).fit()

    def fit_sarima_model(self, model, train_y, order_arima):
        return model(train_y, order=order_arima, seasonal_order=(P, D, Q, m), trend='c')

    def fit_auto_arima_model(self, model, train_y, order_arima):
        d = order_arima[1]
        auto_sarima_model = model(train_y,
                                  start_p=1, start_q=1,
                                  test='adf',
                                  max_p=4, max_q=4, m=7,
                                  start_P=0, seasonal=True,
                                  d=d, D=1,
                                  trace=False,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)
        return auto_sarima_model

    def fit_xgboost_model(self, model, train_X, train_y, test_X, test_y):
        X_train = train_X[self.features]
        X_test = test_X[self.features]
        return model.fit(X_train, train_y, eval_set=[(X_train, train_y), (X_test, test_y)], verbose=False)

    def fit_linear_regression_model(self, model, train_X, train_y):
        return model.fit(train_X, train_y)

    def split_train_test_data(self, data, train_ratio):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    def make_final_prediction(self):
        self.final_submission

