from main import app
from fastapi.testclient import TestClient
import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)


client = TestClient(app)


def test_select_model():
    # Define el cuerpo de la solicitud como un diccionario Python
    request_body = {
        "weights": {
            "RMSE": 0,
            "MAE": 0,
            "R2": 0,
            "MAPE": 0,
            "SMAPE": 1
        },
        "train_models": ["XGBoost", "LinearRegression", "RandomForestRegressor"],
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
        "/training/select-model?type_storage=LOCAL&path_base=buckets/expense-control-bucket", json=request_body)

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
