import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from data_processing import load_data, rename_columns, feature_engineering
from autoML import pycaret_setup, MLSystem
from data_output import pipeline_test, submitt_test



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email ':['moris.jose.jr@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_kaggle',
    default_args=default_args,
    description='Un pipeline para resolver una competencia de kaggle',
    schedule_interval='0 17 * * *',
)

load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

rename_columns = PythonOperator(
    task_id='rename_columns',
    python_callable=rename_columns,
    dag=dag,
)

feature_engineering = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

pycaret_setup = PythonOperator(
    task_id='pycaret_setup',
    python_callable=pycaret_setup,
    dag=dag,
)


modelAutoML = PythonOperator(
    task_id = 'modelAutoML',
    python_callable = MLSystem().crecion_model,
    dag=dag,
)

pipeline_test = PythonOperator(
    task_id='pipeline_test',
    python_callable=pipeline_test,
    dag=dag,
)


submitt_test = PythonOperator(
    task_id='submitt_test',
    python_callable=submitt_test,
    dag=dag,
)


load_data >> rename_columns >> feature_engineering >> pycaret_setup >> modelAutoML >> pipeline_test >> submitt_test