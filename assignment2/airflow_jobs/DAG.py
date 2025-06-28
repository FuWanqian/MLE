import sys
import os

sys.path.append("/opt/airflow")


from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'Group 3',
    'depends_on_past': False,
    'retries': 0
}

with DAG(
    dag_id="Assign2-DAG",
    start_date=datetime(2025, 6, 21),
    schedule_interval=None,
    catchup=False,
    default_args=default_args
) as dag:
   
        model_input = BashOperator(
        task_id="model_input",
        bash_command="env PYTHONPATH=/opt/airflow python /opt/airflow/model_input.py"
        )

        # Model training task
        model_train = BashOperator(
            task_id="model_train",
            bash_command="env PYTHONPATH=/opt/airflow python /opt/airflow/model_train.py"
        )
        
        # Model inference task
        model_inference = BashOperator(
            task_id="model_inference",
            bash_command="env PYTHONPATH=/opt/airflow python /opt/airflow/model_inference.py"
        )
        
        # Model monitoring task
        model_monitor = BashOperator(
            task_id="model_monitor",
            bash_command="env PYTHONPATH=/opt/airflow python /opt/airflow/model_monitor.py"
        )
   
    
        model_input >> model_train >> model_inference >> model_monitor



    