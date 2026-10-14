from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)


# ============================================
# 1. Start Spark
# ============================================

spark = SparkSession.builder \
    .appName("BreastCancerDecisionTree") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========================================")
print(" BREAST CANCER CLASSIFICATION")
print(" Apache Spark MLlib - Decision Tree")
print("========================================")


# ============================================
# 2. Load Dataset
# ============================================

file_path = r"D:\SparkMLProject\breast_cancer.csv"

data = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("\nDataset loaded successfully!")
print("Total records:", data.count())


# ============================================
# 3. Convert Diagnosis into Labels
# ============================================

# Benign = 0
# Malignant = 1

data = data.withColumn(
    "label",
    when(col("diagnosis") == "M", 1.0).otherwise(0.0)
)

feature_columns = [
    c for c in data.columns
    if c not in ["diagnosis", "label"]
]


# ============================================
# 4. Prepare Features
# ============================================

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)


# ============================================
# 5. Split Dataset
# ============================================

train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training records:", train_data.count())
print("Testing records:", test_data.count())


# ============================================
# 6. Create Decision Tree
# ============================================

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

pipeline = Pipeline(
    stages=[assembler, dt]
)


# ============================================
# 7. Train Model
# ============================================

print("\nTraining Decision Tree model...")

model = pipeline.fit(train_data)

print("Decision Tree training completed!")


# ============================================
# 8. Generate Predictions
# ============================================

predictions = model.transform(test_data)

print("\nSample predictions:")

predictions.select(
    "diagnosis",
    "label",
    "prediction"
).show(10)


# ============================================
# 9. Evaluate Model
# ============================================

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


# ============================================
# 10. Display Results
# ============================================

print("\n========================================")
print(" DECISION TREE PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# ============================================
# 11. Display Feature Importance
# ============================================

decision_tree = model.stages[-1]

print("\nTop 10 Feature Importances:")

importance_values = list(
    zip(feature_columns, decision_tree.featureImportances.toArray())
)

importance_values.sort(
    key=lambda item: item[1],
    reverse=True
)

for feature, importance in importance_values[:10]:
    print(f"{feature}: {importance:.4f}")


# ============================================
# 12. Stop Spark
# ============================================

spark.stop()

print("\nDecision Tree evaluation completed!")
print("Apache Spark session closed.")
