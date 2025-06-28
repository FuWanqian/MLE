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

# model inference
y_train_pred = loaded_model.predict(x_train_nor)
y_train_prob = loaded_model.predict_proba(x_train_nor)[:, 1]


print("Train Accuracy:", accuracy_score(y_train_arr, y_train_pred))
print("Train ROC AUC:", roc_auc_score(y_train_arr, y_train_prob))
print("Train Confusion Matrix:\n", confusion_matrix(y_train_arr, y_train_pred))
print("Train Classification Report:\n", classification_report(y_train_arr, y_train_pred))

y_pred = loaded_model.predict(x_test_nor)
y_prob = loaded_model.predict_proba(x_test_nor)[:, 1] 

print(" Accuracy:", accuracy_score(y_test_arr, y_pred))
print(" ROC AUC:", roc_auc_score(y_test_arr, y_prob))
print(" Confusion Matrix:\n", confusion_matrix(y_test_arr, y_pred))
print(" Classification Report:\n", classification_report(y_test_arr, y_pred))


# save results
save_dir = "/opt/airflow/datamart/gold/model_inference"
os.makedirs(save_dir, exist_ok=True)

df_train_pred = pd.DataFrame({
    "y_train_pred": y_train_pred,
    "y_train_prob": y_train_prob
})
df_train_pred.to_csv(os.path.join(save_dir, "train_predictions.csv"), index=False)


df_test_pred = pd.DataFrame({
    "y_test_pred": y_pred,
    "y_test_prob": y_prob
})
df_test_pred.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
