def process_bronze_table(filenames, input_dir, output_dir, start_date, end_date, spark):
    
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