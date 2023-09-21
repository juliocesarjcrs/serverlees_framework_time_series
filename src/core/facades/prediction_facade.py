import pandas as pd
import ast
from src.core.storage.s3Client.s3_client import S3Manager
from src.core.training.time_series_feature_engineering import TimeSeriesFeatureEngineering


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

    def predict(self, params_model: dict):
        """
        Perform a prediction using the loaded model.

        Args:
            data (dict): The input data for prediction.

        Returns:
            The prediction result.
        """
        # Define the start date
        start_date = '2022-11-30'  # Mes siguiente al último entrenado

        # Define the number of months
        months_to_predict = int(params_model['months_to_predict'])
        column_order = params_model['train_columns']
        column_order = ast.literal_eval(column_order)

        # Convert the start date to a datetime object
        start_date = pd.to_datetime(start_date)

        # Generate consecutive dates
        dates = pd.date_range(
            start=start_date, periods=months_to_predict, freq='M')

        # Convert the dates to a list of strings with the desired format
        formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]

        index = pd.to_datetime(formatted_dates)

        data_evaluate = pd.DataFrame(index=index)
        time_series_feature_engineering = TimeSeriesFeatureEngineering()
        data_evaluate, _ = time_series_feature_engineering.feature_engineering_time_series_dynamic(
            data_evaluate)
        # Add holidays
        data_evaluate = time_series_feature_engineering.add_monthly_holiday_count(
            data_evaluate, 'holidays_cont')
        # column_order2 = ['holidays_cont', 'month', 'year', 'days_in_month',
        #                  'is_first_month', 'is_last_month', 'day', 'day_of_week', 'day_of_year']

    # Reorganiza las columnas según el orden deseado
        data_evaluate_ordered = data_evaluate[column_order]

        prediction_result = self.model.predict(data_evaluate_ordered)
        predict_list = prediction_result.tolist()
        prediction = dict(zip(formatted_dates, predict_list))
        data_list = []

        for date, prediction in prediction.items():
            data_list.append({"date": date, "prediction": prediction})

        # Luego, puedes crear la respuesta JSON
        response_data = {'predict': data_list}
        return response_data
