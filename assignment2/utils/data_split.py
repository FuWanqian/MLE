import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta



def split_oot(X_df, y_df, config):
    """
    Split data into OOT and train-test based on snapshot_date and customer_id linkage
    """
    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) &
                      (y_df['snapshot_date'] <= config['model_train_date'])].copy()
    X_model_df = X_df[X_df['customer_id'].isin(y_model_df['customer_id'].unique())].copy()

    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) &
                       (y_model_df['snapshot_date'] <= config['oot_end_date'])].copy()
    X_oot = X_model_df[X_model_df['customer_id'].isin(y_oot['customer_id'].unique())].copy()

    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']].copy()
    X_traintest = X_model_df[X_model_df['customer_id'].isin(y_traintest['customer_id'].unique())].copy()

    return X_traintest, y_traintest, X_oot, y_oot



