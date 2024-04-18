import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from pycaret.regression import *
from kaggle.api.kaggle_api_extended import KaggleApi

from automl_2 import MLSystem


# Inicializa el sistema de ML
system = MLSystem()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 17),
    'email ':['alinare3681@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Abalone_AutoML',
    default_args=default_args,
    description='Abalone Dataset AutoML',
    schedule_interval=timedelta(days=1),
)

def load_data():

    system.load_data()




def train_model(ti):

    train,test,submission=system.preprocess_data()
    model1_path, model2_path, model3_path=system.train_model(train)

    system.create_submission_file(test,submission, model1_path, model2_path, model3_path)


def submit_results(ti):
    print("Finish")
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(file_name='/opt/airflow/dags/data/submission_final.csv',
    message="First submission",
    competition="playground-series-s4e4")



get_data_task = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=load_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=train_model,
    dag=dag,
)

submit_results_task = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=submit_results,
    dag=dag,
)

get_data_task >> train_model_task >> submit_results_task