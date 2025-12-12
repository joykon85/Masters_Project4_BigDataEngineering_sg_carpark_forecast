
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

import os

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Start Spark session

spark = SparkSession.builder.appName("GBTRegressor") \
    .config("spark.driver.memory", "128g") \
    .getOrCreate()

# === Step 1: Load NumPy arrays ===
X_train = np.load("X_train.npy")  # shape (573516, 5, 5)
y_train = np.load("y_train.npy")  # shape (573516,)
X_test = np.load("X_test.npy")    # shape (143379, 5, 5)
y_test = np.load("y_test.npy")    # shape (143379,)

# === Step 2: Flatten 5x5 matrices to 25-element vectors ===
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# === Step 3: Convert to pandas DataFrame ===
feature_cols = [f"f{i}" for i in range(X_train_flat.shape[1])]

df_train_pd = pd.DataFrame(X_train_flat, columns=feature_cols)
df_train_pd["label"] = y_train

df_test_pd = pd.DataFrame(X_test_flat, columns=feature_cols)
df_test_pd["label"] = y_test

# === Step 4: Convert to Spark DataFrame ===
df_train = spark.createDataFrame(df_train_pd)
df_test = spark.createDataFrame(df_test_pd)

# === Step 5: Assemble features ===
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_prepared = assembler.transform(df_train)
test_prepared = assembler.transform(df_test)


# === Step 6: Train GBT Regressor ===
gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter = 250, maxDepth = 10, stepSize = 0.01)
model = gbt.fit(train_prepared)



# === Step 7: Predict and evaluate ===
predictions = model.transform(test_prepared)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"RMSE on test set: {rmse:.4f}")

predictions.select("label", "prediction") \
    .toPandas() \
    .to_csv("predictions_GBT_25Apr.csv", index=False)

model.write().overwrite().save("gbt_model_25Apr")

# Stop the Spark Session
spark.stop()

