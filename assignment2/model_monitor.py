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


# import data
base_dir = "/opt/airflow/datamart/model_input"


x_train_nor = np.load(os.path.join(base_dir, "x_train_nor.npy"))
x_test_nor = np.load(os.path.join(base_dir, "x_test_nor.npy"))
x_oot_nor = np.load(os.path.join(base_dir, "x_oot_nor.npy"))


y_train_arr = np.load(os.path.join(base_dir, "y_train_arr.npy"))
y_test_arr = np.load(os.path.join(base_dir, "y_test_arr.npy"))
y_oot_arr = np.load(os.path.join(base_dir, "y_oot_arr.npy"))


#import model
with open("/opt/airflow/models/logisticmodel.pkl", "rb") as f:
    loaded_model = pickle.load(f)


# model monitor
y_oot_pred = loaded_model.predict(x_oot_nor)
y_oot_prob = loaded_model.predict_proba(x_oot_nor)[:, 1]


print("OOT Accuracy:", accuracy_score(y_oot_arr, y_oot_pred))
print("OOT ROC AUC:", roc_auc_score(y_oot_arr, y_oot_prob))
print("OOT Confusion Matrix:\n", confusion_matrix(y_oot_arr, y_oot_pred))
print("OOT Classification Report:\n", classification_report(y_oot_arr, y_oot_pred))


# save results
monitor_dir = "/opt/airflow/datamart/gold/model_monitor"
os.makedirs(monitor_dir, exist_ok=True)

df_oot_pred = pd.DataFrame({
    "y_oot_pred": y_oot_pred,
    "y_oot_prob": y_oot_prob
})
df_oot_pred.to_csv(os.path.join(monitor_dir, "oot_predictions.csv"), index=False)

