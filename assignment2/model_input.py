# === Standard Library ===
import glob
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyspark
from pyspark.sql import SparkSession

# === Sklearn ===
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    fbeta_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Custom Modules ===
from utils.prepare_data import import_feature_and_label
from utils.data_split import split_oot


# 1. set up spark
spark = SparkSession.builder \
    .appName("ModelInputPrep") \
    .config("spark.driver.memory", "16g")\
    .config("spark.executor.memory", "16g")\
    .config("spark.driver.maxResultSize", "4g")\
    .config("spark.network.timeout", "600s")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .getOrCreate()

# 2. prepare data for model train

# 2.1 import gold data
x,y = import_feature_and_label('/opt/airflow/datamart/gold', spark)


# 2.2set up data split config
model_train_date_str = "2024-09-01"  
train_test_period_months = 12
oot_period_months = 2
train_test_ratio = 0.8

config = {}
config["model_train_date_str"] = model_train_date_str
config["train_test_period_months"] = train_test_period_months
config["oot_period_months"] =  oot_period_months
config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
config["train_test_ratio"] = train_test_ratio

# 2.3 split data
x_traintest, y_traintest, x_oot, y_oot = split_oot(x, y, config)  # split OOT


x_train, x_test, y_train, y_test = train_test_split(x_traintest, y_traintest, 
                                                    test_size=config['train_test_ratio'], 
                                                    random_state=611, 
                                                    shuffle=True, 
                                                    stratify=y_traintest['label'])



#2.4 change data format
# Transform data into numpy arrays
x_train_arr = x_train.drop(columns=['customer_id', 'snapshot_date']).values
x_test_arr = x_test.drop(columns=['customer_id', 'snapshot_date']).values
x_oot_arr = x_oot.drop(columns=['customer_id', 'snapshot_date']).values

y_train_arr = y_train['label'].values
y_test_arr = y_test['label'].values
y_oot_arr = y_oot['label'].values

# Normalize x
scaler = StandardScaler()
x_train_nor = scaler.fit_transform(x_train_arr)
x_test_nor = scaler.transform(x_test_arr)
x_oot_nor = scaler.transform(x_oot_arr)


# 3. save data

save_dir = "/opt/airflow/datamart/model_input"
os.makedirs(save_dir, exist_ok=True)


np.save(os.path.join(save_dir, "x_train_nor.npy"), x_train_nor)
np.save(os.path.join(save_dir, "x_test_nor.npy"), x_test_nor)
np.save(os.path.join(save_dir, "x_oot_nor.npy"), x_oot_nor)


np.save(os.path.join(save_dir, "y_train_arr.npy"), y_train_arr)
np.save(os.path.join(save_dir, "y_test_arr.npy"), y_test_arr)
np.save(os.path.join(save_dir, "y_oot_arr.npy"), y_oot_arr)