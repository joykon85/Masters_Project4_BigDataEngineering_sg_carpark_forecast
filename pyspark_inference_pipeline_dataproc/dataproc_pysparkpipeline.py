
import os
import pickle

from datetime import datetime
from pyspark.sql.types import TimestampType

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

import sys
from pyspark.sql.functions import to_json, struct, col, lit
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import round
from pyspark.sql.types import IntegerType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressionModel

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.sql.functions import lit
from pyspark.sql import Row
from pyspark.sql.functions import max as spark_max, expr

spark = SparkSession.builder.appName("carpark_prediction_pipeline").getOrCreate()

my_project_id = "involuted-forge-456406-a1"  # this is my projectid
bigquery_dataset_input = "smart-car-park-availability-1.lta_data.view_carpark_availability_for_prediction"
bigquery_dataset_output = "smart-car-park-availability-1.lta_data.carpark_predictions_30min_1"

temp_bucket = "gs://dataproc-temp-us-central1-773964370565-z5wnonnu/"
pkl_file_paths = ["gs://carpark_predict_models/4april2025/label_encoder_Carparks.pkl", "gs://carpark_predict_models/4april2025/min_max_scaler_AvailableLots.pkl" ,"gs://carpark_predict_models/4april2025/min_max_scaler_Carparks.pkl", "gs://carpark_predict_models/4april2025/min_max_scaler_Min_Hour_Day.pkl"]
model_dir = "gs://carpark_predict_models/GBT_15Apr_First"
output_dir = "gs://carpark_predict_temp/spark_predictions"

# DECLARE SEQUENCE LENGTH CONSTANT
####################################################
SEQUENCE_LENGTH = 5

##### PART ONE: LOAD AND CHECK #################################################################################################################

# FUNCTIONS USED
####################################################
def check_value_counts_equal_pyspark(df, col_name):
    """
    Checks if all value counts in a PySpark DataFrame column are equal.
    """
    counts = df.groupBy(col_name).count()
    first_count = counts.collect()[0]['count']  # Get the first count
    return counts.filter(F.col('count') != first_count).count() == 0  # Check if any counts differ

# GET FILE FROM BIGQUERY
###################################################
df = spark.read.format("bigquery") \
    .option("table", bigquery_dataset_input) \
    .option("viewsEnabled", "true") \
    .option("temporaryGcsBucket", temp_bucket) \
    .load() \
    .filter(F.col("timestamp") > F.current_timestamp() + F.expr("INTERVAL 454 MINUTES"))

# SORT AND CHECK the data
#############################################################
df = df.orderBy("CarParkID", "timestamp")

if not check_value_counts_equal_pyspark(df, 'CarParkID'):
    #print("CarParkID is not equal. Don't continue. Trigger an error")
    raise RuntimeError("CarParkID is not equal. Aborting the job.")
else:
    #print("CarParkEncoded is equal!")
    counts = df.groupBy('CarParkID').count()
    if counts.agg(F.max('count')).collect()[0][0] != SEQUENCE_LENGTH:
        #print("Don't have Sequence Length of Data Input. Don't continue. Trigger an error")
        raise RuntimeError("Don't have Sequence Length of Data Input. Aborting the job.")

#    else:
        #print("Have right amount of data to proceed. Data Preprocessing Completed Successfully")


##### PART TWO: PREPROCESS #################################################################################################################

# FUNCTIONS USED
####################################################
def create_sequences_per_location_pyspark(df, sequence_length=5):
    """
    Creates sequences from a PySpark DataFrame.
    This is a simplified version; efficient windowing in PySpark is complex.
    """
    # Define the columns to use for the 3D array
    selected_columns = ['CarParkEncoded_MM', 'Min_MM', 'Hour_MM', 'DayOfWeek_MM', 'AvailableLots_MM']

    # Group by 'CarParkEncoded' and collect into lists
#    grouped_df = df.groupBy('CarParkEncoded').agg(F.collect_list(F.array(*selected_columns)).alias('sequences'))
    grouped_df = df.groupBy('CarParkID').agg(F.collect_list(F.array(*selected_columns)).alias('sequences'))

    def check_sequence_length(sequences):
        for seq in sequences:
            if len(seq) < sequence_length:
                return False
        return True

    # Create a UDF to check sequence lengths
    check_length_udf = F.udf(check_sequence_length, BooleanType())

    # Check if all sequences have the correct length
    sequence_done_flag = grouped_df.withColumn('all_sequences_valid', check_length_udf(F.col('sequences'))).select(F.max('all_sequences_valid')).collect()[0][0]

    if not sequence_done_flag:
        return None, False
    else:
        return grouped_df, True

def load_pickles_to_driver(pkl_file_paths):
    """
    Loads pickle files to the driver node.
    WARNING: This is suitable for small pickle files only!
    For large files, distribute them or use Spark's broadcast.
    """
    loaded_objects = []
    from google.cloud import storage
    client = storage.Client()
    for pkl_path in pkl_file_paths:
        bucket_name = pkl_path.split('/')[2]
        blob_name = '/'.join(pkl_path.split('/')[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob_data = blob.download_as_bytes()
        loaded_objects.append(pickle.loads(blob_data))
    return loaded_objects

# OPEN MODEL SCALARS AND ENCODERS
#####################################################
le, scaler_AL, scalar_CP, scaler_temp = load_pickles_to_driver(pkl_file_paths)

# DATA PREPROCESSING
#####################################################
df = df.withColumn('timestamp', F.to_timestamp('timestamp'))

# FEATURE EXTRACTION/ ENCODING/ NORMALISATION
# Extract features
df = df.withColumn('Min', F.minute('timestamp')) \
       .withColumn('Hour', F.hour('timestamp')) \
       .withColumn('DayOfWeek', F.dayofweek('timestamp') - 1) # Adjust to match Pandas (0-6)

# Encode CarParkID (UDF for complex logic)
def encode_carpark_udf(carpark_id):
    try:
        return int(le.transform([carpark_id])[0])
    except ValueError:
        return -1  # Or handle unknown IDs as needed

encode_carpark_spark_udf = F.udf(encode_carpark_udf, IntegerType())
df = df.withColumn('CarParkEncoded', encode_carpark_spark_udf('CarParkID'))

# Sort for time sequence modeling
df = df.orderBy('CarParkEncoded', 'timestamp')

# NORMALISE (UDFs for scaling)
def scale_al_udf(al):
    return float(scaler_AL.transform([[al]])[0][0])

scale_al_spark_udf = F.udf(scale_al_udf, FloatType())
df = df.withColumn('AvailableLots_MM', scale_al_spark_udf('AvailableLots'))

def scale_cp_udf(cp):
    return float(scalar_CP.transform([[cp]])[0][0])

scale_cp_spark_udf = F.udf(scale_cp_udf, FloatType())
df = df.withColumn('CarParkEncoded_MM', scale_cp_spark_udf('CarParkEncoded'))

def scale_time_udf(min, hour, dayofweek):
    scaled_values = scaler_temp.transform([[min, hour, dayofweek]])[0]
    return scaled_values.tolist()

scale_time_spark_udf = F.udf(scale_time_udf, ArrayType(FloatType()))
df = df.withColumn('scaled_time', scale_time_spark_udf('Min', 'Hour', 'DayOfWeek'))
df = df.withColumn('Min_MM', F.col('scaled_time').getItem(0)) \
       .withColumn('Hour_MM', F.col('scaled_time').getItem(1)) \
       .withColumn('DayOfWeek_MM', F.col('scaled_time').getItem(2))

# LSTM SEQUENCING
######################################################
model_input, sequence_done_flag = create_sequences_per_location_pyspark(df, SEQUENCE_LENGTH)

# Extra check. If fail don't proceed to predictions. Need to add some code here
if not sequence_done_flag:
    raise RuntimeError("Not every Carpark has Sequence Length entries. Aborting the job.")
#    print("ERROR! Not every Carpark has Sequence Length entries. Raise Error")
#else:
#    print("Sequencing Done!")

model_input = model_input.orderBy("CarParkID")


##### PART THREE: PREDICT #################################################################################################################

# === Step 1: "Load from Parquet" ===
df_cp = model_input.select("CarParkID")
df_seq = model_input.select("sequences")

# LOAD MODEL
model = GBTRegressionModel.load(model_dir)

# PREPARE THE DATA
# Step 1: Flatten 5x5 array into 1D list (25 elements)
def flatten_array(arr):
    return [item for sublist in arr for item in sublist]

flatten_udf = udf(flatten_array, ArrayType(FloatType()))
df_flat = df_seq.withColumn("flat_features", flatten_udf("sequences"))

# Step 2: Convert to DenseVector
def to_vector(arr):
    return Vectors.dense(arr)

vector_udf = udf(to_vector, VectorUDT())
df_vectorized = df_flat.withColumn("features", vector_udf("flat_features"))

# GET PREDICTIONS
predictions = model.transform(df_vectorized)

# SCALE THE PREDCITIONS

# Step 1: Create a UDF that uses scaler_AL.inverse_transform
def inverse_prediction(pred):
    return float(scaler_AL.inverse_transform([[pred]])[0][0])

# Step 2: Register it as a UDF
inverse_udf = udf(inverse_prediction, FloatType())

# Step 3: Apply it to the prediction column
df_preds_all = predictions.withColumn("PredictedLots", inverse_udf("prediction"))

# Show the result
#df_preds_all.select("prediction", "prediction_original_scale").show()

# GET LAST SET OF ORIGINAL DATA
# Step 1: Get the latest timestamp
max_ts = df.select(F.max("timestamp")).collect()[0][0]

# Step 2: Get the Min value corresponding to that timestamp
min_val_at_latest = df.filter(F.col("timestamp") == F.lit(max_ts)) \
                      .select(F.min("Min")) \
                      .collect()[0][0]

# Step 3: Filter all rows where Min equals that value and select only rows i want
result_df = df.filter(F.col("Min") == F.lit(min_val_at_latest)).select("CarParkID", "AvailableLots", "timestamp", "Location", "Development")

# ADD THE PREDICTIONS IN, CAST TO INT AND ROUND, AND ARRANGE FOR PREDICTION BIGQUERY TABLE
# Add index to both dataframes
result_df_indexed = result_df.withColumn("row_id", monotonically_increasing_id())
df_preds_indexed = df_preds_all.withColumn("row_id", monotonically_increasing_id())

# Join on the row_id
joined_df = result_df_indexed.join(df_preds_indexed.select("row_id", "PredictedLots"), on="row_id").drop("row_id")

final_df = joined_df.select(
    "CarParkID",
    "AvailableLots",
    round(col("PredictedLots")).cast(IntegerType()).alias("PredictedLots"),
    col("timestamp").alias("LatestTimestamp"),
    "Location",
    "Development"
)

# Show the final result
#final_df.show()

# WRITE TO BIG QUERY!
final_df.write \
    .format("bigquery") \
    .option("table", bigquery_dataset_output) \
    .option("temporaryGcsBucket", temp_bucket) \
    .mode("overwrite") \
    .save()

# END SESSION!
spark.stop()

