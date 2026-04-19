import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 1. DATA INGESTION
# =============================================================================
# Loading the industrial sensor dataset from the data directory
try:
    df = pd.read_csv('../01_Data/predictive_maintenance.csv')
except FileNotFoundError:
    print("Execution Error: 'predictive_maintenance.csv' not found in 01_Data.")
    exit()

# =============================================================================
# 2. DATA PRE-PROCESSING & CLEANING
# =============================================================================
# Standardizing column names by removing special characters and spaces
df.columns = df.columns.str.replace(' ', '_').str.replace('[', '').str.replace(']', '')

# Defining the target variable (failure indicator)
target_col = 'failure'

# Displaying initial data structure and types
print("\n🔍 DATASET EXPLORATION:")
print(df.head())
print(df.info())

# =============================================================================
# 3. DIRECTORY MANAGEMENT
# =============================================================================
# Ensuring output directories exist for asset storage
for folder in ['../04_Screenshots', '../05_Output']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Failure Distribution Analysis
print("\n📊 CLASS DISTRIBUTION (0: Normal, 1: Failure):")
print(df[target_col].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x=target_col, data=df, palette='viridis', hue=target_col, legend=False)
plt.title('Machine Failure Distribution')
plt.savefig('../04_Screenshots/failure_distribution.png')
plt.close()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Industrial Sensor Correlation Matrix')
plt.savefig('../04_Screenshots/correlation_matrix.png')
plt.close()

# Statistical profiling during equipment failure
print("\n⚠️ MEAN SENSOR VALUES DURING FAILURE EVENTS:")
failure_analysis = df[df[target_col] == 1].mean(numeric_only=True)
print(failure_analysis)
failure_analysis.to_csv('../05_Output/failure_event_profiles.csv')

# =============================================================================
# 5. PREDICTIVE MODELING (MACHINE LEARNING)
# =============================================================================
# Feature Selection: Dropping target and non-predictive ID/Date columns
cols_to_drop = [target_col, 'device', 'date', 'UDI', 'Product_ID']
X = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
y = df[target_col]

# Encoding categorical features if present (e.g., machine type)
X = pd.get_dummies(X, drop_first=True)

# Stratified Splitting to handle imbalanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initializing and training Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =============================================================================
# 6. FEATURE IMPORTANCE & MODEL EVALUATION
# =============================================================================
# Identifying key sensors driving the predictive model
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importance.plot(kind='barh', color='teal')
plt.title('Primary Sensor Indicators for Machine Failure')
plt.xlabel('Significance Score')
plt.tight_layout()
plt.savefig('../04_Screenshots/feature_importance.png')
plt.close()

# Exporting feature rankings
importance.to_csv('../05_Output/sensor_importance_ranking.csv')

# Model Accuracy Scoring
accuracy = model.score(X_test, y_test)
print("\n🤖 MODEL PERFORMANCE METRICS:")
print(f"Prediction Accuracy: {accuracy:.2%}")

# =============================================================================
# 7. FINAL STATUS
# =============================================================================
print("\n✅ Predictive Maintenance Pipeline Completed Successfully.")
print("📁 CSV Analytics exported to: 05_Output")
print("📸 Visual Assets exported to: 04_Screenshots")