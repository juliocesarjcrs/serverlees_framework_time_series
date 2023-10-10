from fastapi.testclient import TestClient
import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from main import app

# Crea un cliente de prueba para la aplicación
client = TestClient(app)

path = '/preprocessing/generate-processed-data'


def test_invalid_generate_processed_data_request():
    # Prueba la ruta de predicción
    response = client.get(path)
    assert response.status_code == 422


def test_valid_generate_processed_data_request():
    # Prueba la ruta de predicción
    response = client.get(
        f"{path}?type_storage=LOCAL&path_base=buckets/expense-control-bucket&anomaly_detection_method=DBSCAN")
    assert response.status_code == 200
    # Verifica la estructura de la respuesta
    assert "anomalies" in response.json()
    assert isinstance(response.json()["anomalies"], list)

    for anomaly in response.json()["anomalies"]:
        assert "date" in anomaly
        assert "cost" in anomaly
