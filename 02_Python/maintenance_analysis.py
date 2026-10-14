
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    PrecisionRecallDisplay,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "01_Data" / "predictive_maintenance.csv"
SCREENSHOTS_DIR = BASE_DIR / "04_Screenshots"
OUTPUT_DIR = BASE_DIR / "05_Output"

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATA INGESTION AND QUALITY CHECKS
# ============================================================

df = pd.read_csv(DATA_PATH)

df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_", regex=False)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
)

print("\nDATA QUALITY CHECKS")
print("Original dataset shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

sensor_columns = [f"metric{i}" for i in range(1, 10)]
required_columns = ["date", "device", "failure"] + sensor_columns

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df["date"] = pd.to_datetime(
    df["date"],
    format="%m/%d/%Y",
    errors="coerce"
)

for col in sensor_columns + ["failure"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[required_columns].isna().any().any():
    raise ValueError("Missing or invalid required values detected.")

if not df["failure"].isin([0, 1]).all():
    raise ValueError("Failure must contain only 0 and 1.")

df = df.drop_duplicates().copy()
df = df.sort_values(["device", "date"]).reset_index(drop=True)

if df.duplicated(["device", "date"]).any():
    raise ValueError("Duplicate device-date records remain.")

print("\nCleaned dataset shape:", df.shape)
print("Unique devices:", df["device"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())


# ============================================================
# 3. CLASS IMBALANCE ANALYSIS
# ============================================================

target_col = "failure"

class_counts = df[target_col].value_counts().sort_index()
failure_rate = df[target_col].mean() * 100

print("\nCLASS DISTRIBUTION")
print(class_counts)
print(f"Failure rate: {failure_rate:.4f}%")

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target_col,
    hue=target_col,
    palette="viridis",
    legend=False
)

plt.title("Machine Failure Distribution")
plt.xlabel("Failure (0 = Normal, 1 = Failure)")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "failure_distribution.png",
    dpi=300
)
plt.close()


# ============================================================
# 4. SENSOR CORRELATION ANALYSIS
# ============================================================

plt.figure(figsize=(11, 8))

correlation_data = df[sensor_columns + [target_col]]

sns.heatmap(
    correlation_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Industrial Sensor Correlation Matrix")
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "correlation_matrix.png",
    dpi=300
)
plt.close()


# ============================================================
# 5. FAILURE EVENT PROFILING
# ============================================================

failure_events = df[df[target_col] == 1]

failure_profile = failure_events[
    sensor_columns + [target_col]
].mean()

print("\nMEAN SENSOR VALUES DURING FAILURE EVENTS")
print(failure_profile)

# Preserve the original export format for Tableau.
failure_profile.to_csv(
    OUTPUT_DIR / "failure_event_profiles.csv"
)


# ============================================================
# 6. FEATURE SELECTION
# ============================================================

# Only sensor measurements are used as predictors.
# Device identifiers and dates are excluded.

X = df[sensor_columns]
y = df[target_col]
groups = df["device"]


# ============================================================
# 7. GROUP-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split: approximately 80% development and 20% test.
# Devices must not overlap between datasets.

test_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

development_idx, test_idx = next(
    test_splitter.split(X, y, groups=groups)
)

development_X = X.iloc[development_idx]
development_y = y.iloc[development_idx]
development_groups = groups.iloc[development_idx]

# Second split: approximately 60% training
# and 20% validation of the overall devices.

validation_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=123
)

train_relative_idx, val_relative_idx = next(
    validation_splitter.split(
        development_X,
        development_y,
        groups=development_groups
    )
)

train_idx = development_idx[train_relative_idx]
val_idx = development_idx[val_relative_idx]

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]

train_devices = set(groups.iloc[train_idx])
val_devices = set(groups.iloc[val_idx])
test_devices = set(groups.iloc[test_idx])

assert train_devices.isdisjoint(val_devices)
assert train_devices.isdisjoint(test_devices)
assert val_devices.isdisjoint(test_devices)

print("\nGROUP-BASED TRAIN / VALIDATION / TEST SPLIT")

for name, features, labels, devices in [
    ("Training", X_train, y_train, train_devices),
    ("Validation", X_val, y_val, val_devices),
    ("Testing", X_test, y_test, test_devices)
]:
    print(
        f"{name}: {len(features)} records, "
        f"{len(devices)} devices, "
        f"{int(labels.sum())} failures"
    )

if any(
    labels.nunique() < 2
    for labels in [y_train, y_val, y_test]
):
    raise ValueError(
        "Each split must contain both normal and failure records."
    )


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Classification threshold selected using validation F1-score.
selected_threshold = 0.10

rf_probabilities = model.predict_proba(X_test)[:, 1]
rf_predictions = (
    rf_probabilities >= selected_threshold
).astype(int)

# Validation probabilities for threshold analysis.
val_probabilities = model.predict_proba(X_val)[:, 1]


# ============================================================
# 9. DUMMY CLASSIFIER BASELINE
# ============================================================

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_probabilities = baseline.predict_proba(X_test)[:, 1]


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

def evaluate_model(name, actual, predicted, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1_Score": f1_score(
            actual, predicted, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            actual, probabilities
        ),
        "PR_AUC": average_precision_score(
            actual, probabilities
        )
    }


evaluation_df = pd.DataFrame([
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
        rf_probabilities
    ),
    evaluate_model(
        "Dummy Classifier",
        y_test,
        baseline_predictions,
        baseline_probabilities
    )
])

print("\nMODEL PERFORMANCE COMPARISON (TEST, THRESHOLD 0.10)")
print(evaluation_df.round(4).to_string(index=False))

print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)

evaluation_df.to_csv(
    OUTPUT_DIR / "model_evaluation_metrics.csv",
    index=False
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX RESULTS (TEST, THRESHOLD 0.10)")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"]
)

plt.title("Random Forest Confusion Matrix (Threshold 0.10)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "confusion_matrix.png",
    dpi=300
)
plt.close()

pd.DataFrame({
    "Metric": [
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ],
    "Count": [tn, fp, fn, tp]
}).to_csv(
    OUTPUT_DIR / "confusion_matrix_results.csv",
    index=False
)


# ============================================================
# 12. PRECISION-RECALL CURVE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    rf_probabilities,
    name="Random Forest",
    ax=ax
)

ax.axhline(
    y=y_test.mean(),
    color="gray",
    linestyle="--",
    label="Test-set failure rate"
)

ax.set_title("Precision-Recall Curve | Failure Detection")
ax.legend()
fig.tight_layout()

fig.savefig(
    SCREENSHOTS_DIR / "precision_recall_curve.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

print("\nSENSOR FEATURE IMPORTANCE")
print(importance.sort_values(ascending=False))

plt.figure(figsize=(10, 6))

importance.plot(
    kind="barh",
    color="teal"
)

plt.title("Random Forest Sensor Feature Importance")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "feature_importance.png",
    dpi=300
)
plt.close()

# Preserve the original export format for Tableau.
importance.to_csv(
    OUTPUT_DIR / "sensor_importance_ranking.csv"
)


# ============================================================
# 14. TEST PREDICTIONS EXPORT
# ============================================================

predictions_df = df.iloc[test_idx][
    ["date", "device", "failure"]
].copy()

predictions_df["Predicted_Failure"] = rf_predictions
predictions_df["Failure_Probability"] = rf_probabilities

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 15. VALIDATION THRESHOLD ANALYSIS
# ============================================================

# Threshold selection is explored on validation data,
# not on the independent test set.

threshold_results = []

thresholds = [0.50, 0.30, 0.20, 0.10, 0.05, 0.01]

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn_val, fp_val, fn_val, tp_val = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "True_Positives": tp_val,
        "False_Positives": fp_val,
        "False_Negatives": fn_val
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nVALIDATION THRESHOLD COMPARISON")
print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False
)


# ============================================================
# 16. FINAL STATUS
# ============================================================

print("\nPREDICTIVE MAINTENANCE ANALYSIS COMPLETE")
print("Model evaluation results exported to 05_Output.")
print("Visualizations saved to 04_Screenshots.")
print("Threshold comparison uses validation data only.")
print("Test classification metrics use threshold 0.10.")
