from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

import csv
import os

# Create Spark session
spark = SparkSession.builder \
    .appName("BreastCancerHyperparameterTuning") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 50)
print(" BREAST CANCER CLASSIFICATION")
print(" HYPERPARAMETER TUNING - SPARK MLLIB")
print("=" * 50)

try:
    # Load dataset
    file_path = r"D:\SparkMLProject\breast_cancer.csv"

    data = spark.read.csv(
        file_path,
        header=True,
        inferSchema=True
    )

    print("\nDataset loaded successfully!")
    print("Total records:", data.count())

    # Convert diagnosis into numerical labels
    data = data.withColumn(
        "label",
        when(col("diagnosis") == "M", 1.0).otherwise(0.0)
    )

    # Select feature columns
    feature_columns = [
        c for c in data.columns
        if c not in ["diagnosis", "label"]
    ]

    # Convert features into a vector
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="raw_features"
    )

    # Standardize features
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    # Logistic Regression classifier
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100
    )

    # Create pipeline
    pipeline = Pipeline(
        stages=[assembler, scaler, lr]
    )

    # Define hyperparameter combinations
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    print("\nHyperparameter combinations:", len(param_grid))
    print("Cross-validation folds: 3")

    # ROC-AUC evaluator
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    # Configure cross-validation
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=auc_evaluator,
        numFolds=3,
        seed=42,
        parallelism=2
    )

    # Split dataset
    train_data, test_data = data.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("\nTraining records:", train_data.count())
    print("Testing records:", test_data.count())

    # Perform hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    tuned_model = cross_validator.fit(train_data)

    print("Hyperparameter tuning completed!")

    # Generate test predictions
    predictions = tuned_model.transform(test_data)

    # Calculate evaluation metrics
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

    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    roc_auc = auc_evaluator.evaluate(predictions)

    # Display performance
    print("\n" + "=" * 50)
    print(" TUNED MODEL PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # Display best hyperparameters
    best_lr = tuned_model.bestModel.stages[-1]

    print("\n" + "=" * 50)
    print(" BEST HYPERPARAMETERS")
    print("=" * 50)

    print("regParam:", best_lr.getRegParam())
    print("elasticNetParam:", best_lr.getElasticNetParam())
    print("maxIter:", best_lr.getMaxIter())

    # Display cross-validation scores
    print("\nCROSS-VALIDATION ROC-AUC SCORES")

    for index, score in enumerate(
        tuned_model.avgMetrics,
        start=1
    ):
        print(f"Combination {index}: {score:.4f}")

    # Save evaluation results to CSV
    output_file = os.path.join(
        r"D:\SparkMLProject",
        "tuned_model_results.csv"
    )

    with open(
        output_file,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "regParam",
            "elasticNetParam",
            "maxIter"
        ])

        writer.writerow([
            "Tuned Logistic Regression",
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(roc_auc, 4),
            best_lr.getRegParam(),
            best_lr.getElasticNetParam(),
            best_lr.getMaxIter()
        ])

    print("\nResults saved successfully!")
    print("File location:", output_file)

    print("\nModel evaluation completed successfully!")

finally:
    spark.stop()
    print("Apache Spark session closed.")
