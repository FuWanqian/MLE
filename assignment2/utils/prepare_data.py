import os
import glob
from pyspark.sql import SparkSession

import os
import glob

def import_feature_and_label(gold_db, spark):
    """
    Helper function to import both feature_store and label_store from a gold database,
    and convert them to Pandas DataFrames.
    """
    # --- Read feature_store ---
    folder_path_x = os.path.join(gold_db, 'feature_store')
    files_list_x = glob.glob(os.path.join(folder_path_x, '*'))

    if not files_list_x:
        raise FileNotFoundError(f"No feature files found in {folder_path_x}")
    
   
    X_spark = spark.read.parquet(*files_list_x)

    # --- Read label_store ---
    folder_path_y = os.path.join(gold_db, 'label_store')
    files_list_y = glob.glob(os.path.join(folder_path_y, '*'))

    if not files_list_y:
        raise FileNotFoundError(f"No label files found in {folder_path_y}")
    
    y_spark = spark.read.parquet(*files_list_y)

    # Convert to Pandas
    X_pandas = X_spark.toPandas()
    y_pandas = y_spark.toPandas()

    return X_pandas, y_pandas


