from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys

## Путь к пайплайну (меняем на свой)
PATH_TO_PIPELINE = '/Users/aleksandrzhuravlev/test/ml_pipeline_airflow/'
## Путь к к папке для сохранения результатов работы (меняем на свой)
PATH_TO_RESULT = '/Users/aleksandrzhuravlev/test/Results/'

# Добавляем папку etl в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), PATH_TO_PIPELINE, 'etl'))

#from load_data import load_data
from load_data import load_data_local
from preprocess import preprocess_data
from train_model import train_model
from evaluate import evaluate_model
from save_results import save_results

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    'breast_cancer_pipeline',
    default_args=default_args,
    description='ETL-пайплайн для диагностики рака груди',
    schedule_interval=None,
    start_date=datetime(2025, 6, 13),
    catchup=False,
) as dag:
    load_data_task = PythonOperator(
        task_id='load_data_task', 
        python_callable=load_data_local,
        op_kwargs={
            'input_path' : PATH_TO_PIPELINE +  'data/wdbc.data', 
            'output_path' : PATH_TO_PIPELINE + 'data/data.csv'},
        #op_kwargs={'output_path' : '~/airflow/ml_pipeline_airflow/data/data.csv'}
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_task',
        python_callable=preprocess_data,
        op_kwargs={
            'input_path': PATH_TO_PIPELINE + 'data/data.csv', 
            'output_path_train': PATH_TO_PIPELINE + 'data/processed_data_train.csv', 
            'output_path_test': PATH_TO_PIPELINE + 'data/processed_data_test.csv'},
    )

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=train_model,
        op_kwargs={
            'input_path': PATH_TO_PIPELINE + 'data/processed_data_train.csv', 
            'output_path': PATH_TO_PIPELINE + 'model/model.pkl'},
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_task',
        python_callable=evaluate_model,
        op_kwargs={
            'input_path': PATH_TO_PIPELINE + 'data/processed_data_test.csv', 
            'model_path': PATH_TO_PIPELINE + 'model/model.pkl', 
            'output_path': PATH_TO_PIPELINE + 'metrics/metrics.json'},
    )

    save_results_task = PythonOperator(
        task_id='save_results_task',
        python_callable=save_results,
        op_kwargs={
            'model_path': PATH_TO_PIPELINE + 'model/model.pkl', 
            'metrics_path': PATH_TO_PIPELINE + 'metrics/metrics.json', 
            'output_dir': PATH_TO_RESULT},
    )

    load_data_task >> preprocess_task >> train_task >> evaluate_task >> save_results_task