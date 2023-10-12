from fastapi.testclient import TestClient
from src.main import app

# Crea un cliente de prueba para la aplicación
client = TestClient(app)


def test_invalid_predict_request():
    # Prueba la ruta de predicción
    response = client.get("/predict")
    assert response.status_code == 422


def test_valid_predict_request():
    # Prueba la ruta de predicción
    response = client.get(
        "/predict?type_storage=LOCAL&months_to_predict=3&path_base=buckets/expense-control-bucket/models&name_model=best_model.pkl")
    assert response.status_code == 200
    # Verifica que la respuesta sea un JSON válido
    # assert response.headers['content-type'] == 'application/json; charset=utf-8'

    # Verifica la estructura de la respuesta sin verificar los datos exactos
    expected_structure = {
        "predict": [
            {
                "date": str,
                "prediction": float
            }
        ]
    }

    response_json = response.json()

    # Comprueba que la estructura sea igual
    assert expected_structure.keys() == response_json.keys()
    assert "predict" in response_json

    # Verifica que "predict" sea una lista no vacía
    assert isinstance(response_json["predict"], list)
    assert len(response_json["predict"]) > 0

    # Verifica que cada elemento de "predict" tenga la estructura adecuada
    for prediction in response_json["predict"]:
        assert set(prediction.keys()) == {"date", "prediction"}
        assert isinstance(prediction["date"], str)
        assert isinstance(prediction["prediction"], (int, float))
