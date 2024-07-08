import pandas as pd
import numpy as np
import json
from pycaret.classification import setup, create_model, tune_model, finalize_model, save_model
from joblib import dump



def pycaret_setup(ti):
    
    dfTrain = ti.xcom_pull(task_ids='feature_engineering')
    
    # Leer la segmentacion de columnas
    ruta_columnas = '/opt/airflow/dags/data/inputs/seg_columnas.json'
    
    with open(ruta_columnas) as file:
        dict_columnas = json.load(file)

    setup_data = setup(data=dfTrain, 
                   target=dict_columnas["col_target"], 
                   session_id=123, 
                   train_size=0.7, 
                   ignore_features=['id'],
                   numeric_features = dict_columnas["col_numericas"],
                   ordinal_features = {dict_columnas["col_ordinales"][0]: dfTrain[dict_columnas["col_ordinales"][0]].drop_duplicates().sort_values().tolist()},
                   categorical_features = dict_columnas["col_categoricas"]
                   )
    
    return setup_data


class MLSystem():

    def __init__(ti):
        pass

    def crecion_model(ti):

        pycaret_setup = ti.xcom_pull(task_ids='pycaret_setup')
        model_lgbm = create_model('lightgbm',fold = 10)

        # Random Search para entrenar
        param_dist_lgbm = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5,10],
            'min_samples_split': [2, 5, 10]
        }

        # Perform Random Search
        tuned_model_random_lgbm = tune_model(model_lgbm, custom_grid=param_dist_lgbm, search_library='scikit-learn', search_algorithm='random', n_iter=50)

        # Obteniendo modelo final
        final_model = finalize_model(tuned_model_random_lgbm)

        save_model(final_model, "model_lgbm_autoML")

        return "model_lgbm_autoML"
   

