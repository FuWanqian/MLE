import pyspark
from pyspark.sql.functions import col
from datetime import datetime
import os



# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# set up config
start_date = "2023-01-01"
end_date = "2024-12-01"

# set up path
input_dir = "data/"
output_dir = "datamart/bronze/"

# List of CSV filenames to process
filenames = ["features_financials.csv", "features_attributes.csv", "feature_clickstream.csv"]





def process_bronze_table():
    
    dataframes = {}
    
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        df = spark.read.csv(input_path, header=True, inferSchema=True)

        # Filter records within the specified date range
        df_filtered = df.filter(
            (col("snapshot_date") >= start_date) & (col("snapshot_date") <= end_date)
        )

        # Store filtered DataFrame in dictionary using base filename as the key
        key = filename.replace(".csv", "")
        dataframes[key] = df_filtered

        # Save filtered data to CSV with new filename prefix "bronze_"
        bronze_filename = "bronze_" + filename
        output_path = os.path.join(output_dir, bronze_filename)
        df_filtered.toPandas().to_csv(output_path, index=False)
        print(f"Saved filtered data to {output_path} with {df_filtered.count()} rows.")

    return dataframes



bronze_dir = "datamart/bronze/"


#  Read bronze_features_financials.csv
df_financials = spark.read.csv(os.path.join(bronze_dir, "bronze_features_financials.csv"), header=True, inferSchema=True)
print("bronze_features_financials.csv row count:", df_financials.count())

# Read bronze_features_attributes.csv
df_attributes = spark.read.csv(os.path.join(bronze_dir, "bronze_features_attributes.csv"), header=True, inferSchema=True)
print("bronze_features_attributes.csv row count:", df_attributes.count())

# Read bronze_feature_clickstream.csv
df_clickstream = spark.read.csv(os.path.join(bronze_dir, "bronze_feature_clickstream.csv"), header=True, inferSchema=True)
print("bronze_feature_clickstream.csv row count:", df_clickstream.count())



df_financials = df_financials.dropna()

df_financials = df_financials.filter(
    (col("Changed_Credit_Limit") != "_") & (col("Credit_Mix") != "_")
)

df_financials = df_financials.filter(
    col("Payment_Behaviour") != "!@9#%8"
)

df_attributes = df_attributes.filter(col("Occupation") != "_______")

from pyspark.sql.functions import col, when, regexp_replace

for c in df_financials.columns:
    df_financials = df_financials.withColumn(
        c,
        when(
            col(c).cast("string").rlike("_$"),  
            regexp_replace(col(c).cast("string"), "_$", "")  
        ).otherwise(col(c))
    )


for c in df_attributes.columns:
    df_attributes = df_attributes.withColumn(
        c,
        when(
            col(c).cast("string").rlike("_$"),
            regexp_replace(col(c).cast("string"), "_$", "")
        ).otherwise(col(c))
    )


    from pyspark.sql.functions import col

# Remove rows where any of the following numeric columns have negative values
df_financials = df_financials.filter(
    (col("Num_Bank_Accounts") >= 0) &
    (col("Num_of_Loan") >= 0) &
    (col("Delay_from_due_date") >= 0) &
    (col("Num_of_Delayed_Payment") >= 0)
)

# Remove rows where Payment_of_Min_Amount is 'NM'
df_financials = df_financials.filter(col("Payment_of_Min_Amount") != "NM")

df_attributes = df_attributes.filter((col("Age") >= 0) & (col("Age") <= 122))


from pyspark.sql.functions import col, regexp_extract, when
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DateType

# 1. Convert snapshot_date to DateType
df_financials = df_financials.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

# 2. Convert Credit_History_Age from "X Years and Y Months" to float
df_financials = df_financials.withColumn("Credit_History_Years", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast("int"))
df_financials = df_financials.withColumn("Credit_History_Months", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast("int"))

df_financials = df_financials.withColumn(
    "Credit_History_Age_Num",
    (col("Credit_History_Years") + col("Credit_History_Months") / 12).cast("double")
)

df_financials = df_financials.drop("Credit_History_Years", "Credit_History_Months", "Credit_History_Age")
df_financials = df_financials.withColumnRenamed("Credit_History_Age_Num", "Credit_History_Age")

# 3. Columns to treat as categorical (keep as string)
categorical_cols = ["Customer_ID", "Type_of_Loan", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "snapshot_date", "Credit_History_Age"]

# 4. Convert all other columns to double
for c, dtype in df_financials.dtypes:
    if c not in categorical_cols:
        df_financials = df_financials.withColumn(c, col(c).cast(DoubleType()))


# 1. Convert Age to double
df_attributes = df_attributes.withColumn("Age", col("Age").cast(DoubleType()))

# 2. Convert snapshot_date to date
df_attributes = df_attributes.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

# 3. Convert all other columns to string (except Age and snapshot_date)
for c in df_attributes.columns:
    if c not in ["Age", "snapshot_date"]:
        df_attributes = df_attributes.withColumn(c, col(c).cast("string"))


# 1. Convert snapshot_date to DateType
df_clickstream = df_clickstream.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

# 2. Ensure Customer_ID is string (in case it's numeric)
df_clickstream.withColumn("Customer_ID", col("Customer_ID").cast("string"))


from scipy.stats import linregress

from pyspark.sql.functions import col, avg, stddev

agg_exprs = []
for c in df_clickstream.columns:
    if c.startswith("fe_"):
        agg_exprs.append(avg(col(c)).alias(c + "_mean"))
        agg_exprs.append(stddev(col(c)).alias(c + "_std"))

df_clickstream_stats = df_clickstream.groupBy("Customer_ID").agg(*agg_exprs)




# Save as single CSV using pandas
df_financials.toPandas().to_csv("datamart/silver/silver_features_financials.csv", index=False)
df_attributes.toPandas().to_csv("datamart/silver/silver_features_attributes.csv", index=False)
df_clickstream_stats.toPandas().to_csv("datamart/silver/silver_feature_clickstream.csv", index=False)


# Define silver directory
silver_dir = "datamart/silver/"

# Read silver_features_financials.csv
df_financials = spark.read.csv(os.path.join(silver_dir, "silver_features_financials.csv"), header=True, inferSchema=True)
print("silver_features_financials.csv row count:", df_financials.count())

# Read silver_features_attributes.csv
df_attributes = spark.read.csv(os.path.join(silver_dir, "silver_features_attributes.csv"), header=True, inferSchema=True)
print("silver_features_attributes.csv row count:", df_attributes.count())

# Read silver_feature_clickstream.csv
df_clickstream_stats = spark.read.csv(os.path.join(silver_dir, "silver_feature_clickstream.csv"), header=True, inferSchema=True)
print("silver_feature_clickstream.csv row count:", df_clickstream_stats.count())

df_joined = df_financials.join(df_attributes, on="Customer_ID", how="inner")
df_joined = df_joined.join(df_clickstream_stats, on="Customer_ID", how="inner")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_joined_pd = df_joined.toPandas()

# Set seaborn style
sns.set(style="whitegrid")

# ---- Age  ----
plt.figure(figsize=(10, 5))
sns.histplot(df_joined_pd ["Age"], bins=30, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Set plot style
sns.set(style="whitegrid")

# Plot all Occupation categories
plt.figure(figsize=(14, 8))
occupation_counts = df_joined_pd ["Occupation"].value_counts()
sns.barplot(x=occupation_counts.index, y=occupation_counts.values, palette="mako")

plt.title("Occupation Distribution (All Categories)")
plt.xlabel("Occupation")
plt.ylabel("Frequency")
plt.xticks(rotation=75, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()



# Set plot style
sns.set(style="whitegrid")

# List of financial ability features
features = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Monthly_Balance",
    "Amount_invested_monthly"
]

# Plot distribution for each feature
for feature in features:
    plt.figure(figsize=(10, 4))
    sns.histplot(df_joined_pd [feature], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


upper = df_joined_pd["Annual_Income"].quantile(0.99)
plt.figure(figsize=(10, 5))
sns.histplot(df_joined_pd[df_joined_pd["Annual_Income"] <= upper]["Annual_Income"],
             bins=30, kde=True, color="salmon")
plt.title("Annual_Income (Capped at 99th Percentile)")
plt.xlabel("Annual_Income")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


features = [
    "Num_Bank_Accounts", 
    "Num_Credit_Card", 
    "Num_of_Loan", 
    "Outstanding_Debt", 
    "Total_EMI_per_month", 
    "Credit_Utilization_Ratio"
]

# Set seaborn style
sns.set(style="whitegrid")

# Loop through each feature and plot
for feature in features:
    plt.figure(figsize=(10, 4))
    sns.histplot(df_joined_pd[feature], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Set seaborn style
sns.set(style="whitegrid")

# Loop through fe_1_mean to fe_20_mean
for i in range(1, 21):
    col = f"fe_{i}_mean"
    if col in df_joined_pd.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_joined_pd[col], bins=30, kde=True, color="lightcoral")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


# Select all fe_*_mean columns
fe_mean_cols = [f"fe_{i}_mean" for i in range(1, 21)]

# Subset DataFrame
df_fe_mean = df_joined_pd[fe_mean_cols].dropna()

# Compute correlation matrix
corr_matrix = df_fe_mean.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap: fe_1_mean to fe_20_mean")
plt.tight_layout()
plt.show()


# Define silver directory
silver_dir = "datamart/silver/"

# Read silver_features_financials.csv
df_financials = spark.read.csv(os.path.join(silver_dir, "silver_features_financials.csv"), header=True, inferSchema=True)
print("silver_features_financials.csv row count:", df_financials.count())

# Read silver_features_attributes.csv
df_attributes = spark.read.csv(os.path.join(silver_dir, "silver_features_attributes.csv"), header=True, inferSchema=True)
print("silver_features_attributes.csv row count:", df_attributes.count())

# Read silver_feature_clickstream.csv
df_clickstream_stats = spark.read.csv(os.path.join(silver_dir, "silver_feature_clickstream.csv"), header=True, inferSchema=True)
print("silver_feature_clickstream.csv row count:", df_clickstream_stats.count())


df_joined = df_financials.join(df_attributes, on="Customer_ID", how="inner")
df_joined = df_joined.join(df_clickstream_stats, on="Customer_ID", how="inner")


# Drop columns in Spark DataFrame
df_joined = df_joined.drop("SSN", "snapshot_date", "Type_of_Loan")


# Convert to Pandas
df_joined_pd = df_joined.toPandas()

# Save as CSV
df_joined_pd.to_csv("datamart/gold/gold_features.csv", index=False)