import json
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

# --------------------------------------------------
# 1. Start Apache Spark
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Breast Cancer Classification") \
    .config("spark.hadoop.fs.permissions.umask-mode", "000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========================================")
print(" BREAST CANCER CLASSIFICATION")
print(" Apache Spark MLlib - Logistic Regression")
print("========================================\n")


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

file_path = r"D:\SparkMLProject\breast_cancer.csv"

data = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Dataset loaded successfully!")
print("Total records:", data.count())
print("Total columns:", len(data.columns))

print("\nDataset preview:")
data.show(5, truncate=False)


# --------------------------------------------------
# 3. Convert diagnosis into numerical labels
# --------------------------------------------------
# B = Benign = 0
# M = Malignant = 1

data = data.withColumn(
    "label",
    when(col("diagnosis") == "M", 1.0).otherwise(0.0)
)

feature_columns = [
    column for column in data.columns
    if column not in ["diagnosis", "label"]
]


# --------------------------------------------------
# 4. Prepare feature vectors
# --------------------------------------------------

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

# --------------------------------------------------
# 5. Split dataset
# --------------------------------------------------

train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining records:", train_data.count())
print("Testing records:", test_data.count())


# --------------------------------------------------
# 6. Create Logistic Regression model
# --------------------------------------------------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

pipeline = Pipeline(
    stages=[assembler, scaler, lr]
)


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model = pipeline.fit(train_data)

# Export trained model parameters for live prediction

os.makedirs(r"D:\SparkMLProject", exist_ok=True)

scaler_model = model.stages[1]
lr_model = model.stages[2]

model_parameters = {
    "feature_names": feature_columns,
    "scaler_mean": scaler_model.mean.toArray().tolist(),
    "scaler_std": scaler_model.std.toArray().tolist(),
    "coefficients": lr_model.coefficients.toArray().tolist(),
    "intercept": float(lr_model.intercept),
    "threshold": float(lr_model.getThreshold())
}

with open(
    r"D:\SparkMLProject\model_parameters.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(model_parameters, file, indent=4)

print("Model parameters exported successfully!")

print("\nTrained Model Pipeline Stages:")

for stage in model.stages:
    print(type(stage).__name__)

# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

predictions = model.transform(test_data)

print("\nSample predictions:")
predictions.select(
    "diagnosis",
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)


# --------------------------------------------------
# 9. Evaluate model performance
# --------------------------------------------------

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)
auc = auc_evaluator.evaluate(predictions)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n========================================")
print(" LOGISTIC REGRESSION PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nModel evaluation completed successfully!")


# --------------------------------------------------
# 11. Stop Spark
# --------------------------------------------------

spark.stop()

print("\nApache Spark session closed.")
