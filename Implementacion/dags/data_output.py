import pandas as pd
import numpy as np
from joblib import load
import json
from pycaret.classification import load_model, predict_model


def pipeline_test(ti):
    
    model_path = ti.xcom_pull(task_ids='modelAutoML')
    
    # Leer nombres de columnas
    ruta_nombres = '/opt/airflow/dags/data/inputs/dict_nombres.json'
    with open(ruta_nombres) as file:
        dict_nombres = json.load(file)

    dfTest = pd.read_csv('/opt/airflow/dags/data/inputs/test.csv')

    # Cambiar nombres de columna
    dfTest.rename(columns = dict_nombres, inplace = True) 
    
    # Leer la segmentacion de columnas
    ruta_columnas = '/opt/airflow/dags/data/inputs/seg_columnas.json'
    with open(ruta_columnas) as file:
        dict_columnas = json.load(file)

    # Crear dummies
    for col in dict_columnas["col_categoricas"]:
        dfDummies = pd.get_dummies(dfTest[col], prefix = col).astype(int)
        dfTest = pd.concat([dfTest, dfDummies], axis = 1)
    
    dfTest['est_civil_relacion'] = dfTest.apply(lambda row: 1 if row['est_civil'] in (2,5) else 0, axis = 1)

    # Leer el modelo 
    modelAutoML = load_model(model_path)

    # Predecir
    dfPredict = predict_model(modelAutoML, data=dfTest)

    return dfPredict

    
def submitt_test(ti):
    dfPredict = ti.xcom_pull(task_ids='modelAutoML')
    
    # Generando df output
    result = dfPredict[['id','prediction_label']].rename(columns = {'prediction_label':'Target'})

    result.to_csv( '/opt/airflow/dags/data/outputs/submission_final.csv', index = False)

    return result
