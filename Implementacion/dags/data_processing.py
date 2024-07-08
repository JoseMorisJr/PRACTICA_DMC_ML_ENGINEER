import pandas as pd
import numpy as np
import json
import os
# from kaggle.api.kaggle_api_extended import KaggleApi
# import zipfile

# # Parametrias para descargar información
# with open('./kaggle.json') as file:
#     credentialsKaggle = json.load(file)

# os.environ['KAGGLE_USERNAME'] = credentialsKaggle['username']
# os.environ['KAGGLE_KEY'] = credentialsKaggle['key']



# def Descarga_informacion():
#     api = KaggleApi()
#     api.authenticate()

#     # Download the competition files
#     competition_name = 'playground-series-s4e6'
#     download_path = './data/inputs'
#     api.competition_download_files(competition_name, path=download_path)

#     # Unzip the downloaded files
#     for item in os.listdir(download_path):
#         if item.endswith('.zip'):
#             zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
#             zip_ref.extractall(download_path)
#             zip_ref.close()
#             print(f"Unzipped {item}")


# Carga de información

def load_data():
    dfTrain = pd.read_csv('/opt/airflow/dags/data/inputs/train.csv')
    return dfTrain

def rename_columns(ti):
    dfTrain = ti.xcom_pull(task_ids='load_data')
    
    # Leer nombres de columnas
    ruta_nombres = '/opt/airflow/dags/data/inputs/dict_nombres.json'
    with open(ruta_nombres) as file:
        dict_nombres = json.load(file)

    # Cambiar nombres de columna
    dfTrain.rename(columns = dict_nombres, inplace = True) 
    return dfTrain

def feature_engineering(ti):
    dfTrain = ti.xcom_pull(task_ids='rename_columns')
    
    # Leer la segmentacion de columnas
    ruta_columnas = '/opt/airflow/dags/data/inputs/seg_columnas.json'
    with open(ruta_columnas) as file:
        dict_columnas = json.load(file)

    # Crear dummies
    for col in dict_columnas["col_categoricas"]:
        dfDummies = pd.get_dummies(dfTrain[col], prefix = col).astype(int)
        dfTrain = pd.concat([dfTrain, dfDummies], axis = 1)
    
    dfTrain['est_civil_relacion'] = dfTrain.apply(lambda row: 1 if row['est_civil'] in (2,5) else 0, axis = 1)
    
    return dfTrain




