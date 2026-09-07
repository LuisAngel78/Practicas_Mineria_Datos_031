import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Reemplaza con los datos de tu archivo kaggle.json
os.environ['KAGGLE_USERNAME'] = "el nombre de usuario de kaggle"
os.environ['KAGGLE_KEY'] = "una api key generada en el perfil de kaggle para poder descargar el dataset"

api = KaggleApi()
api.authenticate()

dataset_id = "rohitsahoo/sales-forecasting"
api.dataset_download_files(dataset_id, path="./datos", unzip=True)