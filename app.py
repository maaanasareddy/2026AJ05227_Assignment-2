"""
Breast Cancer Classification - Streamlit App
Implements 5 ML models with evaluation metrics and confusion matrix display.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

# --- Page Config ---
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# --- Title ---
st.title("Breast Cancer Classification Dashboard")
st.markdown("""
This application demonstrates **5 Machine Learning classification models** trained on the 
[Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

**Models implemented:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Support Vector Machine

---
""")

# --- Helper Functions ---
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

FEATURE_COLS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]


@st.cache_resource
def load_model(model_name):
    """Load a saved model from disk."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the saved StandardScaler."""
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@st.cache_resource
def load_label_encoder():
    """Load the saved LabelEncoder."""
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC Score": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "MCC Score": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# --- Sidebar ---
st.sidebar.header("Configuration")

# Dataset upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"],
    help="Upload a CSV file containing the test data with features and 'diagnosis' column."
)

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys()),
    index=0
)

show_all_models = st.sidebar.checkbox("Compare All Models", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** Breast Cancer Wisconsin (Diagnostic)\n\n"
    "**Target:** diagnosis (M=Malignant, B=Benign)\n\n"
    "**Features:** 30 numeric features"
)

# --- Main Content ---
# Load artifacts
scaler = load_scaler()
le = load_label_encoder()

# Process data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"Uploaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
else:
    data = pd.read_csv("test_data.csv")
    st.info("Using default test data (test_data.csv). Upload your own CSV from the sidebar.")

# Display dataset preview
with st.expander("Preview Dataset", expanded=False):
    st.dataframe(data.head(20), width="stretch")
    st.write(f"**Shape:** {data.shape[0]} rows x {data.shape[1]} columns")

# Validate data
missing_cols = [c for c in FEATURE_COLS if c not in data.columns]
if missing_cols:
    st.error(f"Missing required feature columns: {missing_cols}")
    st.stop()

if "diagnosis" not in data.columns:
    st.error("Missing 'diagnosis' column in the uploaded data.")
    st.stop()

# Prepare features and target
X = data[FEATURE_COLS]
y_raw = data["diagnosis"]

# Encode target if needed
if not pd.api.types.is_numeric_dtype(y_raw):
    y = le.transform(y_raw)
else:
    y = y_raw.values

X_scaled = scaler.transform(X)

# --- Single Model Evaluation ---
if not show_all_models:
    st.header(f"Model: {selected_model}")

    model = load_model(selected_model)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    metrics = compute_metrics(y, y_pred, y_prob)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    col6.metric("MCC Score", f"{metrics['MCC Score']:.4f}")

    st.markdown("---")

    # Visualizations
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y, y_pred, le.classes_)
        st.pyplot(fig_cm)

    with col_roc:
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y, y_prob)
        st.pyplot(fig_roc)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), width="stretch")

# --- Compare All Models ---
else:
    st.header("Model Comparison")

    all_results = {}
    for name in MODEL_FILES:
        model = load_model(name)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        all_results[name] = compute_metrics(y, y_pred, y_prob)

    # Comparison table
    comparison_df = pd.DataFrame(all_results).T
    comparison_df.index.name = "Model"

    st.subheader("Evaluation Metrics Comparison")
    st.dataframe(
        comparison_df.style
        .highlight_max(axis=0, color="lightgreen")
        .format("{:.4f}"),
        width="stretch"
    )

    # Bar chart comparison
    st.subheader("Metrics Comparison Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0.7, 1.05)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Confusion matrices for all models
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for idx, name in enumerate(MODEL_FILES):
        model = load_model(name)
        y_pred = model.predict(X_scaled)
        with cols[idx % 3]:
            st.markdown(f"**{name}**")
            fig_cm = plot_confusion_matrix(y, y_pred, le.classes_)
            st.pyplot(fig_cm)

    # ROC curves overlay
    st.subheader("ROC Curves (All Models)")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (name, color) in zip(MODEL_FILES, colors):
        model = load_model(name)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("ROC Curves - All Models", fontsize=14)
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_roc)

    # Best model
    best = max(all_results, key=lambda k: all_results[k]["F1 Score"])
    st.success(f"**Best Model (by F1 Score):** {best} with F1 = {all_results[best]['F1 Score']:.4f}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "*Breast Cancer Classification App | M.Tech AIML - Machine Learning Assignment 2 | "
    "Built with Streamlit & scikit-learn*"
)
