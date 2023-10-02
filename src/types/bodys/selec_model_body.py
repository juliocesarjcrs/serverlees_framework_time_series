"""
Este módulo define modelos Pydantic para representar pesos y el cuerpo de una solicitud 'SelectModel'.
"""
from pydantic import BaseModel
from src.enums.models_type import ModelsType
from src.enums.dataset_type import DatasetType
class Weights(BaseModel):
    """
    Modelo Pydantic para representar un conjunto de pesos.

    Args:
        RMSE (float): Valor para la métrica RMSE.
        MAE (float): Valor para la métrica MAE.
        R2 (float): Valor para la métrica R^2.
        MAPE (float): Valor para la métrica MAPE.
        SMAPE (float): Valor para la métrica SMAPE.
    """

    RMSE: float
    MAE: float
    R2: float
    MAPE: float
    SMAPE: float

class datasetToTrain(BaseModel):
    """
    Modelo Pydantic para representar un conjunto de pesos.

    Args:
        name (DatasetType): Valor para la métrica RMSE.
        path (string): Valor para la métrica MAE.
    """

    name: DatasetType
    path: str
class SelectModelBody(BaseModel):
    """
    Modelo Pydantic para representar el cuerpo de una solicitud 'SelectModel'.

    Args:
        weights (Weights): Objeto Weights que contiene los pesos para métricas específicas.
    """

    weights: Weights
    train_models: list[ModelsType]
    datasets_to_train: list[datasetToTrain]
