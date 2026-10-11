import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROJECT CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "breast_cancer.csv")
COMPARISON_PATH = os.path.join(BASE_DIR, "model_comparison.csv")
TUNING_PATH = os.path.join(BASE_DIR, "tuned_model_results.csv")
MODEL_PARAMETERS_PATH = os.path.join(
    BASE_DIR, "model_parameters.json"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1 {
    color: #60a5fa !important;
    font-weight: 800 !important;
}

h2 {
    color: #93c5fd !important;
    font-weight: 700 !important;
}

h3 {
    color: #c4b5fd !important;
}

p, li {
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stMetric"] {
    background-color: #172338;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 30px !important;
    font-weight: bold !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 12px;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
    border: 1px solid #60a5fa;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #334155;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


df = None
comparison_df = None
tuning_df = None

if os.path.exists(DATASET_PATH):
    df = load_csv(DATASET_PATH)

if os.path.exists(COMPARISON_PATH):
    comparison_df = load_csv(COMPARISON_PATH)

if os.path.exists(TUNING_PATH):
    tuning_df = load_csv(TUNING_PATH)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🎗️ Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Home Dashboard",
        "Dataset Overview",
        "Model Performance",
        "Model Comparison",
        "Hyperparameter Tuning",
        "Confusion Matrix",
        "ROC Curve",
        "Live Prediction",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Project\n\n"
    "Built using Python, Apache Spark MLlib, "
    "Pandas, Plotly and Streamlit."
)

# =====================================================
# HOME DASHBOARD
# =====================================================

if page == "Home Dashboard":

    st.title("🎗️ Breast Cancer Classification")

    st.markdown(
        "### Apache Spark MLlib | Machine Learning | Model Evaluation"
    )

    st.write("")

    st.success(
        "Welcome to the Breast Cancer Classification dashboard!"
    )

    st.markdown("## Project Overview")

    st.write(
        "This project demonstrates the use of machine learning "
        "algorithms to classify records in a breast cancer dataset "
        "into benign and malignant categories."
    )

    st.write(
        "Apache Spark MLlib is used for model training, "
        "hyperparameter tuning and performance evaluation."
    )

    st.warning(
        "Educational demonstration only. This application is "
        "not intended for medical diagnosis or clinical decisions."
    )

    st.markdown("## Project Statistics")

    if df is not None:

        total_records = len(df)

        total_features = len([
            column for column in df.columns
            if column.lower() != "diagnosis"
        ])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", total_records)
        col2.metric("Dataset Features", total_features)
        col3.metric("Models Compared", 3)

        # ---------------------------------------------
        # DIAGNOSIS DISTRIBUTION
        # ---------------------------------------------

        if "diagnosis" in df.columns:

            benign_count = int(
                (df["diagnosis"] == "B").sum()
            )

            malignant_count = int(
                (df["diagnosis"] == "M").sum()
            )

            st.markdown("## Diagnosis Distribution")

            # Original bar chart

            distribution = pd.DataFrame({
                "Diagnosis": ["Benign", "Malignant"],
                "Records": [benign_count, malignant_count]
            })

            st.bar_chart(
                distribution.set_index("Diagnosis")
            )

            # Interactive doughnut chart

            st.markdown("### Interactive Distribution Chart")

            diagnosis_counts = pd.DataFrame({
                "Diagnosis": ["Benign", "Malignant"],
                "Count": [benign_count, malignant_count]
            })

            fig = px.pie(
                diagnosis_counts,
                names="Diagnosis",
                values="Count",
                title="Benign vs Malignant Distribution",
                hole=0.45,
                color="Diagnosis",
                color_discrete_map={
                    "Benign": "#38bdf8",
                    "Malignant": "#a78bfa"
                }
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                font_color="#f1f5f9",
                title_x=0.5,
                legend_title_text="Diagnosis"
            )

            fig.update_traces(
                textinfo="label+percent",
                hovertemplate=(
                    "%{label}<br>"
                    "Records: %{value}<br>"
                    "Percentage: %{percent}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.error(
                "The dataset does not contain a diagnosis column."
            )

    else:
        st.error(
            "Dataset not found. Make sure breast_cancer.csv "
            "is inside the project folder."
        )

    # ---------------------------------------------
    # MACHINE LEARNING ALGORITHMS
    # ---------------------------------------------

    st.markdown("## Machine Learning Algorithms")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Logistic Regression")

        st.write(
            "A classification algorithm that estimates "
            "the probability of a class."
        )

    with col2:
        st.subheader("Decision Tree")

        st.write(
            "A tree-based algorithm that makes predictions "
            "using feature-based decisions."
        )

    with col3:
        st.subheader("Hyperparameter Tuning")

        st.write(
            "Cross-validation compares parameter combinations "
            "to help select a model configuration."
        )

# =====================================================
# DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    if df is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.markdown("### Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="breast_cancer_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("### Statistical Summary")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )

        if "diagnosis" in df.columns:

            st.markdown("### Diagnosis Counts")

            counts = (
                df["diagnosis"]
                .value_counts()
                .rename(index={
                    "B": "Benign",
                    "M": "Malignant"
                })
            )

            st.bar_chart(counts)

    else:
        st.error(
            "Dataset file was not found. "
            "Check that breast_cancer.csv exists."
        )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    if comparison_df is not None:

        required_columns = [
            "Model",
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)",
            "ROC-AUC (%)"
        ]

        if all(
            column in comparison_df.columns
            for column in required_columns
        ):

            selected_model = st.selectbox(
                "Choose a model",
                comparison_df["Model"].tolist()
            )

            model_data = comparison_df[
                comparison_df["Model"] == selected_model
            ].iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f'{model_data["Accuracy (%)"]:.2f}%'
            )

            col2.metric(
                "Precision",
                f'{model_data["Precision (%)"]:.2f}%'
            )

            col3.metric(
                "Recall",
                f'{model_data["Recall (%)"]:.2f}%'
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "F1 Score",
                f'{model_data["F1 Score (%)"]:.2f}%'
            )

            col5.metric(
                "ROC-AUC",
                f'{model_data["ROC-AUC (%)"]:.2f}%'
            )

            st.markdown("### Complete Performance Table")

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.error(
                "The comparison CSV does not contain all expected "
                "metric columns. Check model_comparison.csv."
            )

    else:
        st.error(
            "model_comparison.csv was not found."
        )

# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("📈 Model Comparison")

    if comparison_df is not None:

        metric_columns = [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)",
            "ROC-AUC (%)"
        ]

        if all(
            column in comparison_df.columns
            for column in ["Model"] + metric_columns
        ):

            st.write(
                "Compare the evaluation metrics of the three models."
            )

            chart_data = comparison_df.set_index("Model")[
                metric_columns
            ]

            st.bar_chart(chart_data)

            # Interactive model comparison chart

            st.markdown("### Interactive Performance Comparison")

            interactive_data = comparison_df.melt(
                id_vars=["Model"],
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score (%)"
            )

            fig = px.bar(
                interactive_data,
                x="Metric",
                y="Score (%)",
                color="Model",
                barmode="group",
                title="Model Performance Comparison",
                text_auto=".2f",
                color_discrete_sequence=[
                    "#38bdf8",
                    "#a78bfa",
                    "#34d399"
                ]
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                font_color="#f1f5f9",
                yaxis_range=[0, 105],
                legend_title_text="Model"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("### Performance Results")

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="📥 Download Model Comparison (CSV)",
                data=comparison_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="model_comparison_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.info(
                "The tuned Logistic Regression model selected "
                "the same best parameters as the original model. "
                "Therefore, their reported test metrics are identical."
            )

        else:
            st.error(
                "Required metric columns are missing from "
                "model_comparison.csv."
            )

    else:
        st.error(
            "model_comparison.csv was not found."
        )

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

elif page == "Hyperparameter Tuning":

    st.title("🎯 Hyperparameter Tuning")

    st.write(
        "Logistic Regression was tuned using three-fold "
        "cross-validation and ROC-AUC evaluation."
    )

    st.markdown("### Parameters Evaluated")

    tuning_parameters = pd.DataFrame({
        "Parameter": [
            "regParam",
            "elasticNetParam",
            "maxIter",
            "Cross-validation folds"
        ],
        "Configuration": [
            "0.001, 0.01, 0.1",
            "0.0, 0.5, 1.0",
            "100",
            "3"
        ]
    })

    st.dataframe(
        tuning_parameters,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Best Parameters")

    col1, col2, col3 = st.columns(3)

    col1.metric("regParam", "0.01")
    col2.metric("elasticNetParam", "0.0")
    col3.metric("maxIter", "100")

    if tuning_df is not None:

        st.markdown("### Saved Tuning Results")

        st.dataframe(
            tuning_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 Download Tuning Results (CSV)",
            data=tuning_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="hyperparameter_tuning_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "The saved tuning-results CSV was not found. "
            "The best parameters displayed above are from "
            "the previously completed tuning run."
        )

    st.caption(
        "Three-fold cross-validation was used to compare "
        "nine parameter combinations."
    )

# =====================================================
# CONFUSION MATRIX
# =====================================================

elif page == "Confusion Matrix":

    st.title("🔲 Confusion Matrix")

    image_path = os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )

    if os.path.exists(image_path):

        st.image(
            image_path,
            caption="Logistic Regression Confusion Matrix",
            use_container_width=True
        )

        st.markdown("### Test Set Results")

        col1, col2 = st.columns(2)

        col1.metric("True Negatives", 51)
        col2.metric("False Positives", 0)

        col3, col4 = st.columns(2)

        col3.metric("False Negatives", 3)
        col4.metric("True Positives", 32)

        st.markdown("### Interpretation")

        st.write(
            "**True Negative:** A benign record correctly "
            "classified as benign."
        )

        st.write(
            "**False Positive:** A benign record incorrectly "
            "classified as malignant."
        )

        st.write(
            "**False Negative:** A malignant record incorrectly "
            "classified as benign."
        )

        st.write(
            "**True Positive:** A malignant record correctly "
            "classified as malignant."
        )

        st.caption(
            "These are results from the existing 86-record "
            "test run, not a clinical evaluation."
        )

    else:

        st.error(
            "confusion_matrix.png was not found. "
            "Run confusion_matrix.py first."
        )

# =====================================================
# ROC CURVE
# =====================================================

elif page == "ROC Curve":

    st.title("📉 ROC Curve")

    image_path = os.path.join(
        BASE_DIR,
        "roc_curve.png"
    )

    if os.path.exists(image_path):

        st.image(
            image_path,
            caption="Logistic Regression ROC Curve",
            use_container_width=True
        )

        st.metric(
            "ROC-AUC Score",
            "0.9978"
        )

        st.markdown("### What does ROC-AUC mean?")

        st.write(
            "The ROC curve compares the true positive rate "
            "with the false positive rate at different "
            "classification thresholds."
        )

        st.write(
            "AUC summarizes the model's ability to distinguish "
            "between the two classes on the evaluated data."
        )

        st.caption(
            "This score describes the saved experiment results "
            "and does not establish clinical usefulness."
        )

    else:

        st.error(
            "roc_curve.png was not found. "
            "Run roc_curve.py first."
        )

# =====================================================
# LIVE PREDICTION
# =====================================================

elif page == "Live Prediction":

    st.title("🔬 Breast Cancer Classification")
    st.markdown("### Logistic Regression | Apache Spark MLlib")

    st.info(
        "This page demonstrates breast cancer classification "
        "using a trained machine-learning model and numerical "
        "measurements from the Wisconsin Breast Cancer dataset."
    )

    st.warning(
        "Educational prototype only. This model has not been "
        "clinically validated for diagnosing patients. Its output "
        "must not be used to make medical decisions."
    )

    # -------------------------------------------------
    # 1. CHECK MODEL FILE
    # -------------------------------------------------

    if not os.path.exists(MODEL_PARAMETERS_PATH):

        st.error(
            "Model parameters were not found. "
            "Please run logistic_regression_model.py first."
        )

    else:

        try:

            # Load model parameters
            with open(
                MODEL_PARAMETERS_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                model_parameters = json.load(file)

            feature_names = model_parameters["feature_names"]

            scaler_mean = np.asarray(
                model_parameters["scaler_mean"],
                dtype=float
            )

            scaler_std = np.asarray(
                model_parameters["scaler_std"],
                dtype=float
            )

            coefficients = np.asarray(
                model_parameters["coefficients"],
                dtype=float
            )

            intercept = float(
                model_parameters["intercept"]
            )

            threshold = float(
                model_parameters["threshold"]
            )

            # -------------------------------------------------
            # 2. VALIDATE MODEL PARAMETERS
            # -------------------------------------------------

            parameter_lengths = [
                len(feature_names),
                len(scaler_mean),
                len(scaler_std),
                len(coefficients)
            ]

            if len(set(parameter_lengths)) != 1:

                st.error(
                    "The model parameters have inconsistent "
                    "dimensions. Run logistic_regression_model.py "
                    "again to regenerate them."
                )

                st.stop()

            if len(feature_names) == 0:

                st.error("No model features were found.")
                st.stop()

            if (
                not np.all(np.isfinite(scaler_mean))
                or not np.all(np.isfinite(scaler_std))
                or not np.all(np.isfinite(coefficients))
                or not np.isfinite(intercept)
                or not np.isfinite(threshold)
            ):

                st.error(
                    "Invalid numerical values were found "
                    "in the model parameters."
                )

                st.stop()

            if np.any(scaler_std <= 0):

                st.error(
                    "Invalid scaling values were found. "
                    "Please regenerate the model parameters."
                )

                st.stop()

            if not 0 < threshold < 1:

                st.error(
                    "The classification threshold is invalid."
                )

                st.stop()

            st.success(
                "Trained model parameters loaded successfully!"
            )

            # -------------------------------------------------
            # 3. CHOOSE INPUT METHOD
            # -------------------------------------------------

            st.markdown("---")
            st.markdown("## Enter Dataset Measurements")

            st.write(
                "Enter the numerical measurements used by the "
                "trained model. You can enter them manually or "
                "select an existing dataset record."
            )

            df = None

            if os.path.exists(DATASET_PATH):

                try:

                    df = pd.read_csv(DATASET_PATH)

                except (OSError, ValueError, pd.errors.ParserError):

                    st.warning(
                        "The dataset could not be read. "
                        "You can still enter measurements manually."
                    )

            dataset_available = (
                df is not None
                and all(
                    feature in df.columns
                    for feature in feature_names
                )
            )

            if dataset_available:

                input_mode = st.radio(
                    "Choose input method",
                    [
                        "Enter values manually",
                        "Use a sample record from the dataset"
                    ],
                    horizontal=True
                )

            else:

                input_mode = "Enter values manually"

                st.caption(
                    "Dataset sample selection is unavailable. "
                    "You can enter the measurements manually."
                )

            # -------------------------------------------------
            # 4. SELECT DATASET RECORD
            # -------------------------------------------------

            sample_row = None

            if (
                dataset_available
                and input_mode == "Use a sample record from the dataset"
            ):

                sample_index = st.number_input(
                    "Dataset row number (starting from 0)",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    step=1
                )

                sample_row = df.iloc[int(sample_index)]

                st.caption(
                    "The selected row supplies the feature measurements. "
                    "Its diagnosis label is not used to calculate "
                    "the prediction."
                )

            # -------------------------------------------------
            # 5. FEATURE INPUT FORM
            # -------------------------------------------------

            st.markdown("### Feature Inputs")

            input_values = {}

            with st.form("live_prediction_form"):

                for start in range(
                    0,
                    len(feature_names),
                    3
                ):

                    feature_group = feature_names[
                        start:start + 3
                    ]

                    st.markdown(
                        f"**Measurements {start + 1}–"
                        f"{start + len(feature_group)}**"
                    )

                    input_columns = st.columns(
                        len(feature_group)
                    )

                    for column_box, feature in zip(
                        input_columns,
                        feature_group
                    ):

                        if (
                            sample_row is not None
                            and input_mode
                            == "Use a sample record from the dataset"
                        ):

                            default_value = float(
                                sample_row[feature]
                            )

                        else:

                            default_value = 0.0

                        input_values[feature] = (
                            column_box.number_input(
                                feature,
                                value=default_value,
                                format="%.6f",
                                key=f"live_{feature}"
                            )
                        )

                predict_clicked = st.form_submit_button(
                    "🔍 Predict Classification",
                    use_container_width=True
                )

            # -------------------------------------------------
            # 6. PREDICTION
            # -------------------------------------------------

            if predict_clicked:

                values = np.asarray(
                    [
                        input_values[feature]
                        for feature in feature_names
                    ],
                    dtype=float
                )

                if not np.all(np.isfinite(values)):

                    st.error(
                        "Please enter valid numerical measurements."
                    )

                else:

                    # Apply the same standardization parameters
                    # exported from the trained Spark model.

                    scaled_values = (
                        values - scaler_mean
                    ) / scaler_std

                    # Calculate the logistic regression score

                    logit = float(
                        np.dot(
                            scaled_values,
                            coefficients
                        ) + intercept
                    )

                    # Calculate the probability for class M

                    if logit >= 0:

                        probability_malignant = (
                            1.0 / (1.0 + np.exp(-logit))
                        )

                    else:

                        exp_logit = np.exp(logit)

                        probability_malignant = (
                            exp_logit / (1.0 + exp_logit)
                        )

                    probability_benign = (
                        1.0 - probability_malignant
                    )

                    # Apply the exported classification threshold

                    predicted_class = (
                        1
                        if probability_malignant > threshold
                        else 0
                    )

                    # -------------------------------------------------
                    # 7. DISPLAY RESULTS
                    # -------------------------------------------------

                    st.markdown("---")
                    st.markdown("## Prediction Results")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        if predicted_class == 1:

                            st.metric(
                                "Model Classification",
                                "Malignant (M)"
                            )

                        else:

                            st.metric(
                                "Model Classification",
                                "Benign (B)"
                            )

                    with result_col2:

                        st.metric(
                            "Model Score for Malignant Class",
                            f"{probability_malignant * 100:.6f}%"
                        )

                    st.markdown(
                        "### Model Probability Distribution"
                    )

                    probability_df = pd.DataFrame(
                        {
                            "Class": [
                                "Benign (B)",
                                "Malignant (M)"
                            ],
                            "Model Score (%)": [
                                probability_benign * 100,
                                probability_malignant * 100
                            ]
                        }
                    )

                    # Display a bar chart using Streamlit

                    st.bar_chart(
                        probability_df.set_index("Class"),
                        y="Model Score (%)"
                    )

                    # Display both scores as a table

                    st.dataframe(
                        probability_df,
                        hide_index=True,
                        use_container_width=True
                    )

                    # -------------------------------------------------
                    # 8. INTERPRETATION
                    # -------------------------------------------------

                    st.markdown("---")
                    st.markdown("### Interpretation")

                    if predicted_class == 1:

                        st.warning(
                            "The model classified the supplied "
                            "measurements as malignant (M). This is "
                            "a model prediction, not a confirmed "
                            "medical diagnosis."
                        )

                    else:

                        st.success(
                            "The model classified the supplied "
                            "measurements as benign (B). This does "
                            "not rule out cancer and is not a "
                            "confirmed medical diagnosis."
                        )

                    st.markdown(
                        "### Recommended Next Step"
                    )

                    st.write(
                        "A qualified healthcare professional should "
                        "interpret any real patient's findings using "
                        "appropriate clinical examinations and "
                        "diagnostic tests."
                    )

                    st.caption(
                        "This educational research prototype has not "
                        "been established as a clinically validated "
                        "diagnostic system. Model scores are not "
                        "equivalent to a patient's actual probability "
                        "of having cancer. Do not use this result "
                        "independently for medical decisions."
                    )

        except (
            KeyError,
            ValueError,
            TypeError,
            OSError,
            OverflowError
        ) as error:

            st.error(
                f"Could not load or use model parameters: {error}"
            )
                                        
# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.markdown("### Project Title")

    st.write(
        "Breast Cancer Classification Using Apache Spark MLlib "
        "with Hyperparameter Tuning and Performance Evaluation"
    )

    st.markdown("### Project Objective")

    st.write(
        "To demonstrate machine learning classification, "
        "model evaluation and hyperparameter tuning using "
        "Apache Spark MLlib."
    )

    st.markdown("### Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Apache Spark MLlib",
            "Pandas",
            "Streamlit",
            "Plotly",
            "Matplotlib",
            "CSV"
        ],
        "Purpose": [
            "Programming language",
            "Machine learning algorithms",
            "Dataset processing",
            "Interactive dashboard",
            "Interactive visualizations",
            "Evaluation charts",
            "Dataset and results storage"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Algorithms and Methods")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree Classification")
    st.write("3. Cross-validation and hyperparameter tuning")
    st.write("4. Confusion matrix evaluation")
    st.write("5. ROC curve and ROC-AUC evaluation")

    st.markdown("### Project Workflow")

    workflow = [
        "1. Load the breast cancer dataset.",
        "2. Prepare the features and target label.",
        "3. Split the dataset into training and testing sets.",
        "4. Train the classification models.",
        "5. Tune Logistic Regression parameters using cross-validation.",
        "6. Evaluate the models using performance metrics.",
        "7. Display the results in the Streamlit dashboard."
    ]

    for step in workflow:
        st.write(step)

    st.warning(
        "This project is for educational and demonstration "
        "purposes only. It must not be used to diagnose "
        "cancer or make medical decisions."
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Breast Cancer Classification Project | "
    "Python • Apache Spark MLlib • Streamlit • Plotly"
)