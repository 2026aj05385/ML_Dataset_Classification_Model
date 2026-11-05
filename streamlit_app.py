"""
Paddy Productivity Class Explorer  (Streamlit front-end)

Loads the pre-trained pipelines from ./model (no retraining), scores an uploaded
test CSV, and shows metrics, confusion matrix, classification report and a
side-by-side comparison of all five models.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

st.set_page_config(page_title="Paddy Productivity Classifier", page_icon="🌾", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading saved models...")
def load_artifacts():
    meta = joblib.load(MODEL_DIR / "metadata.joblib")
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    return meta, models


def read_csv(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    return df


# ----------------------------------------------------------------------------- metrics
def score_model(model, X, y_true, class_order):
    """Return (metrics dict, predictions, probabilities). Never retrains."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                            labels=list(model.classes_))
    except ValueError:  # e.g. uploaded file lacks one of the classes
        auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred, proba


def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- layout
meta, models = load_artifacts()
TARGET = meta["target_col"]
CLASSES = meta["class_order"]
FEATURES = meta["feature_columns"]
low_cut, high_cut = meta["bin_edges_kg_per_ha"]

st.title("🌾 Paddy Productivity Class Predictor")
st.caption(
    f"Predicts whether a paddy plot's yield per hectare is **Low** (< {low_cut:,.0f} Kg/ha), "
    f"**Medium** ({low_cut:,.0f}–{high_cut:,.0f} Kg/ha) or **High** (> {high_cut:,.0f} Kg/ha) "
    f"from {len(FEATURES)} agronomic and weather features. Five pre-trained models; nothing is retrained here."
)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + target column)", type="csv")
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None,
                             disabled=uploaded is not None)
    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))
    st.markdown("---")
    st.caption(f"Required target column: `{TARGET}` with values {', '.join(CLASSES)}.")
    with st.expander("Required feature columns"):
        st.write(FEATURES)

if uploaded is not None:
    data = read_csv(uploaded)
    source_label = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    data = read_csv(SAMPLE_CSV)
    source_label = "bundled test_data.csv"
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled sample) to begin.")
    st.stop()

# ----------------------------------------------------------------------------- validation
missing_cols = [c for c in FEATURES if c not in data.columns]
if missing_cols:
    st.error(f"The file is missing {len(missing_cols)} required feature column(s): {missing_cols[:8]}"
             f"{' ...' if len(missing_cols) > 8 else ''}")
    st.stop()
if TARGET not in data.columns:
    st.error(f"Target column `{TARGET}` not found. Add it with values from {CLASSES}.")
    st.stop()
bad_labels = sorted(set(data[TARGET].dropna().astype(str)) - set(CLASSES))
if bad_labels:
    st.error(f"Unexpected values in `{TARGET}`: {bad_labels}. Allowed: {CLASSES}.")
    st.stop()
if data[FEATURES].isna().any().any():
    st.warning("Rows with missing feature values were dropped.")
data = data.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
if data.empty:
    st.error("No usable rows left after removing missing values.")
    st.stop()

X = data[FEATURES]
y_true = data[TARGET].astype(str)

# ----------------------------------------------------------------------------- results
all_results, selected = {}, None
for name, mdl in models.items():
    m, pred, proba = score_model(mdl, X, y_true, CLASSES)
    all_results[name] = m
    if name == model_name:
        selected = (m, pred, proba, mdl)
metrics, y_pred, proba, chosen = selected

c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", f"{len(data):,}")
c2.metric("Source", source_label)
c3.metric("Selected model", model_name)

tab_model, tab_compare, tab_preds = st.tabs(
    ["📊 Selected model", "🏁 Compare all 5 models", "🔎 Row-level predictions"])

with tab_model:
    st.subheader(f"{model_name} — evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion(y_true, y_pred, CLASSES, model_name))
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=CLASSES,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))
        never = [c for c in CLASSES if c not in set(y_pred)]
        if never:
            st.warning(f"This model never predicts: {', '.join(never)}.")

with tab_compare:
    st.subheader("All models on the same data")
    comp = pd.DataFrame(all_results).T
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"))
    metric_pick = st.radio("Chart metric", list(comp.columns), horizontal=True, index=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.barplot(x=comp.index, y=comp[metric_pick], hue=comp.index, palette="crest", legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(metric_pick)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"Best on {metric_pick}: **{comp[metric_pick].idxmax()}** "
               f"({comp[metric_pick].max():.4f}). Highlighted cells = column maximum.")

with tab_preds:
    st.subheader(f"Predictions from {model_name}")
    out = data.copy()
    out["Predicted_Class"] = y_pred
    for i, cls in enumerate(chosen.classes_):
        out[f"P({cls})"] = proba[:, i].round(3)
    out["Correct"] = out["Predicted_Class"] == out[TARGET]
    shown = ["Agriblock", "Variety", "Soil Types", "Hectares", TARGET, "Predicted_Class", "Correct"] \
        + [f"P({c})" for c in chosen.classes_]
    st.dataframe(out[shown], height=360)
    st.download_button("⬇ Download full predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")
