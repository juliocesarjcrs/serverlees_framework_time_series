from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
path_url = '/training/select-model'
type_storage = 'LOCAL'
path_base = 'buckets/expense-control-bucket'


def test_select_model():
    train_models = ["XGBoost", "LinearRegression", "RandomForestRegressor"]
    request_body = {
        "weights": {
            "RMSE": 0,
            "MAE": 0,
            "R2": 0,
            "MAPE": 0,
            "SMAPE": 1
        },
        "train_models": train_models,
        "datasets_to_train": [
            {
                "name": "Normal",
                "path": "data/processed/df_time_monthly.csv"
            },
            {
                "name": "Without Outliers",
                "path": "data/processed/df_time_monthly_without_outliers.csv"
            }
        ]
    }

    # Realiza la solicitud POST al endpoint de selección de modelos
    response = client.post(
        f"{path_url}?type_storage={type_storage}&path_base={path_base}", json=request_body)

    # Verifica que la respuesta sea exitosa (código 200) y contiene la estructura esperada
    assert response.status_code == 200

    # Define la estructura esperada de la respuesta
    expected_structure = {
        "model_metrics": [
            {
                "dataset_name": str,
                "model_name": str,
                "metrics": {
                    "MSE": float,
                    "MAE": float,
                    "R2": float,
                    "RMSE": float,
                    "MAPE": float,
                    "SMAPE": float,
                    "MAPE_percent": float
                },
                "parameters_training": {
                    "train_columns": [str]
                },
                "start_date_test": str
            }
        ],
        "best_model": [
            {
                "model_name": str,
                "dataset_name": str,
                "weighted_metric": float
            }
        ]
    }

    # Verifica que la respuesta tenga la estructura esperada sin comprobar los valores específicos
    assert all(key in response.json() for key in expected_structure)


def test_select_model_undefined_invalid_input():
    train_models = ["XGBoost", "LinearRegression",
                    "RandomForestRegressor", "Undefined"]
    request_body = {
        "weights": {
            "RMSE": 0,
            "MAE": 0,
            "R2": 0,
            "MAPE": 0,
            "SMAPE": 1
        },
        "train_models": train_models,
        "datasets_to_train": [
            {
                "name": "Normal",
                "path": "data/processed/df_time_monthly.csv"
            },
            {
                "name": "Without Outliers",
                "path": "data/processed/df_time_monthly_without_outliers.csv"
            }
        ]
    }

    response = client.post(
        f"{path_url}?type_storage={type_storage}&path_base={path_base}", json=request_body)
    assert response.status_code == 400
    expected_response = {
        "detail": "Error en el cuerpo de la solicitud 1 validation error for SelectModelBody\ntrain_models.3\n  Input should be 'Arima','SARIMA','SARIMA-AutoArima','XGBoost','LinearRegression','BoostedHybrid' or 'RandomForestRegressor' [type=enum, input_value='Undefined', input_type=str]"
    }

    assert response.json() == expected_response
