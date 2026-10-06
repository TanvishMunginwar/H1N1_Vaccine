import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="H1N1 Vaccine Prediction", page_icon="💉", layout="wide")

st.title("💉 H1N1 Vaccine Prediction")
st.write("Logistic Regression model based on the preprocessing and training steps from the Jupyter Notebook.")

# -----------------------------
# Preprocessing from notebook
# -----------------------------
def preprocess_data(hn):
    hn = hn.copy()

    if "unique_id" in hn.columns:
        hn = hn.drop(["unique_id"], axis=1)

    # Numeric missing values -> median
    numeric_cols = hn.select_dtypes(include="number").columns
    hn[numeric_cols] = hn[numeric_cols].fillna(hn[numeric_cols].median())

    # Categorical missing values -> same values used in notebook
    fill_values = {
        "qualification": "College Graduate",
        "income_level": "<= $75,000, Above Poverty",
        "marital_status": "Married",
        "housing_status": "Own",
        "employment": "Employed",
    }

    for col, value in fill_values.items():
        if col in hn.columns:
            hn[col] = hn[col].fillna(value)

    # Same manual mappings as notebook
    mappings = {
        "income_level": {
            "Below Poverty": 0,
            "> $75,000": 1,
            "<= $75,000, Above Poverty": 2,
        },
        "qualification": {
            "< 12 Years": 0,
            "12 Years": 1,
            "Some College": 2,
            "College Graduate": 3,
        },
        "census_msa": {
            "MSA, Not Principle  City": 0,
            "MSA, Principle City": 1,
            "Non-MSA": 2,
        },
    }

    for col, mapping in mappings.items():
        if col in hn.columns:
            hn[col] = hn[col].replace(mapping)

    # Same LabelEncoder columns as notebook.
    # We return the encoders so they can also be used for new input data.
    encoder_cols = [
        "age_bracket",
        "race",
        "sex",
        "marital_status",
        "housing_status",
        "employment",
    ]

    encoders = {}
    for col in encoder_cols:
        if col in hn.columns:
            le = LabelEncoder()
            hn[col] = le.fit_transform(hn[col].astype(str))
            encoders[col] = le

    return hn, encoders


def preprocess_new_data(new_df, encoders):
    """Apply the same transformations to new prediction rows."""
    df = new_df.copy()

    if "unique_id" in df.columns:
        df = df.drop(["unique_id"], axis=1)

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    fill_values = {
        "qualification": "College Graduate",
        "income_level": "<= $75,000, Above Poverty",
        "marital_status": "Married",
        "housing_status": "Own",
        "employment": "Employed",
    }
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    mappings = {
        "income_level": {
            "Below Poverty": 0,
            "> $75,000": 1,
            "<= $75,000, Above Poverty": 2,
        },
        "qualification": {
            "< 12 Years": 0,
            "12 Years": 1,
            "Some College": 2,
            "College Graduate": 3,
        },
        "census_msa": {
            "MSA, Not Principle  City": 0,
            "MSA, Principle City": 1,
            "Non-MSA": 2,
        },
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen categories safely.
            values = df[col].astype(str)
            unknown = ~values.isin(le.classes_)
            if unknown.any():
                raise ValueError(
                    f"Unknown value(s) found in '{col}': "
                    f"{values[unknown].unique().tolist()}"
                )
            df[col] = le.transform(values)

    return df


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload the H1N1 CSV file used by your notebook",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload your H1N1 vaccine CSV file from the sidebar to train the model.")
    st.markdown("""
    ### Notebook model used
    - **Algorithm:** Logistic Regression
    - **Target:** `h1n1_vaccine`
    - **Test size:** 20%
    - **Random state:** 132
    - **Training-set duplication:** rows where `h1n1_vaccine == 1` are duplicated, matching the notebook
    - **Missing values:** numeric median + specified categorical defaults
    - **Encoding:** manual mapping + LabelEncoder
    """)
    st.stop()

# -----------------------------
# Load and train
# -----------------------------
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if "h1n1_vaccine" not in raw_df.columns:
    st.error("The uploaded CSV must contain the target column: `h1n1_vaccine`")
    st.stop()

try:
    hn, encoders = preprocess_data(raw_df)

    hn_train, hn_test = train_test_split(
        hn, test_size=0.2, random_state=132
    )

    # Same duplication/oversampling step from notebook
    df0 = hn_train[hn_train.h1n1_vaccine == 1]
    hn_train = pd.concat([hn_train, df0])

    hn_train_x = hn_train.iloc[:, 0:-1]
    hn_train_y = hn_train.iloc[:, -1]

    hn_test_x = hn_test.iloc[:, 0:-1]
    hn_test_y = hn_test.iloc[:, -1]

    lr = LogisticRegression(max_iter=1000)
    lr.fit(hn_train_x, hn_train_y)

    pred = lr.predict(hn_test_x)
    pred_prob = lr.predict_proba(hn_test_x)[:, -1]

except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# -----------------------------
# Metrics
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

accuracy = accuracy_score(hn_test_y, pred) * 100
precision = precision_score(hn_test_y, pred, zero_division=0) * 100
recall = recall_score(hn_test_y, pred, zero_division=0) * 100
f1 = f1_score(hn_test_y, pred, zero_division=0) * 100
roc_auc = roc_auc_score(hn_test_y, pred_prob) * 100

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Precision", f"{precision:.2f}%")
col3.metric("Recall", f"{recall:.2f}%")
col4.metric("F1 Score", f"{f1:.2f}%")
col5.metric("ROC-AUC", f"{roc_auc:.2f}%")

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(hn_test_y, pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)
st.dataframe(cm_df, use_container_width=True)

# -----------------------------
# Dataset overview
# -----------------------------
with st.expander("📋 Dataset Overview"):
    st.write("Original dataset shape:", raw_df.shape)
    st.dataframe(raw_df.head(10), use_container_width=True)

# -----------------------------
# Batch prediction
# -----------------------------
st.subheader("🔮 Batch Prediction")

st.write(
    "Upload a CSV containing the same input columns as the notebook "
    "(the target column is optional for prediction)."
)

prediction_file = st.file_uploader(
    "Upload data for prediction",
    type=["csv"],
    key="prediction_file"
)

if prediction_file is not None:
    try:
        new_raw = pd.read_csv(prediction_file)

        # Target is not required for prediction
        if "h1n1_vaccine" in new_raw.columns:
            new_input = new_raw.drop(columns=["h1n1_vaccine"])
        else:
            new_input = new_raw.copy()

        new_processed = preprocess_new_data(new_input, encoders)

        # Ensure exact training feature order
        expected_features = list(hn_train_x.columns)
        missing = [c for c in expected_features if c not in new_processed.columns]
        extra = [c for c in new_processed.columns if c not in expected_features]

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif extra:
            new_processed = new_processed.drop(columns=extra)

        if not missing:
            new_processed = new_processed[expected_features]

            predictions = lr.predict(new_processed)
            probabilities = lr.predict_proba(new_processed)[:, 1]

            result = new_raw.copy()
            result["Prediction"] = predictions
            result["Probability_of_H1N1_Vaccine"] = probabilities.round(4)
            result["Prediction_Label"] = result["Prediction"].map({
                0: "No",
                1: "Yes"
            })

            st.dataframe(result, use_container_width=True)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv,
                "h1n1_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Built from the preprocessing and Logistic Regression workflow in the provided Jupyter Notebook.")
