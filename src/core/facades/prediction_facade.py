from datetime import datetime
import pandas as pd
import ast
from dateutil.relativedelta import relativedelta
from src.core.training.time_series_feature_engineering import TimeSeriesFeatureEngineering
from src.utils.logger.logger import Logger


class PredictionFacade:
    """
    A facade class for making predictions using pre-trained machine learning models.

    Args:
        model: The pre-trained machine learning model to be used for predictions.

    Attributes:
        model: The loaded machine learning model in the facade.
    """
    model = None

    def __init__(self, model):
        self.model = model
        self.logger = Logger("PredictionFacade")

    def predict(self, params_model: dict):
        """
        Perform a prediction using the loaded model.

        Args:
            data (dict): The input data for prediction.

        Returns:
            The prediction result.
        """
        months_to_predict = int(params_model['months_to_predict'])
        start_date_test = params_model['start_date_test']  # Mes siguiente al último entrenado
        now = datetime.now()
        self.logger.info(f"::: now ::: {now}")
        end_date = now + relativedelta(months=+months_to_predict)

        end_date_format =end_date.date()
        self.logger.info(f"::: end_date ::: {end_date_format}")

        # Define the number of months
        column_order = params_model['train_columns']

        # Convert the start date to a datetime object
        start_date = pd.to_datetime(start_date_test)

        # Generate consecutive dates
        range_dates = pd.date_range(
            start=start_date, end=end_date_format, freq='M')

        # Convert the dates to a list of strings with the desired format
        formatted_dates = [date.strftime('%Y-%m-%d') for date in range_dates]

        index = pd.to_datetime(formatted_dates)

        data_evaluate = pd.DataFrame(index=index)
        time_series_feature_engineering = TimeSeriesFeatureEngineering()
        data_evaluate, _ = time_series_feature_engineering.feature_engineering_time_series_dynamic(
            data_evaluate)
        # Add holidays
        data_evaluate = time_series_feature_engineering.add_monthly_holiday_count(
            data_evaluate, 'holidays_cont')

    # Reorganiza las columnas según el orden deseado
        data_evaluate_ordered = data_evaluate[column_order]

        prediction_result = self.model.predict(data_evaluate_ordered)
        predict_list = prediction_result.tolist()
        prediction = dict(zip(formatted_dates, predict_list))
        data_list = []

        for date_value, prediction in prediction.items():
            data_list.append({"date": date_value, "prediction": prediction})

        # Luego, puedes crear la respuesta JSON
        response_data = {'predict': data_list}
        return response_data
