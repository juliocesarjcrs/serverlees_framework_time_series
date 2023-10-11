import requests

from fastapi import FastAPI
from src.lambdas.forecast.prediction import prediction
from src.lambdas.preprocessing.generate_processed_data import generate_processed_data
from src.lambdas.training.select_model import select_model
from src.lambdas.preprocessing.get_last_data import get_last_data
from src.lambdas.utils.generate_folder_structure import generate_folder_structure

app = FastAPI()

# Monta las rutas definidas en los otros archivos
app.include_router(prediction.router)
app.include_router(generate_processed_data.router)
app.include_router(select_model.router)
app.include_router(get_last_data.router)
app.include_router(generate_folder_structure.router)
