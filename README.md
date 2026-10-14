# Predictive Maintenance - Machine Learning Classification

## Project Overview

This project analyzes industrial equipment sensor data to investigate machine failure patterns and evaluate the performance of a Random Forest classification model.

The analysis focuses on identifying recorded failure events using sensor measurements, with particular attention to class imbalance, data leakage prevention, and model evaluation.

The project combines Python for data analysis and machine learning with Tableau for interactive reporting.

## Business Objectives

- Analyze equipment sensor measurements and failure patterns.
- Identify sensor variables associated with machine failures.
- Develop a classification model to distinguish normal operation from failure events.
- Evaluate model performance using metrics appropriate for imbalanced datasets.
- Compare classification thresholds and assess the trade-off between missed failures and false alarms.
- Present findings through interactive Tableau dashboards.

## Dataset

The dataset contains approximately 124,000 daily equipment observations from 1,169 industrial devices.

The target variable is `failure`, where 0 represents normal operation and 1 represents a recorded failure.

Nine sensor variables (`metric1` to `metric9`) are used as model features.

Only 106 failure events are recorded, representing approximately 0.085% of observations. This substantial class imbalance is a central challenge of the analysis.

## Tools and Technologies

- Python: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn, Random Forest Classifier
- Model Evaluation: Precision, Recall, F1-score, ROC-AUC, Average Precision, Confusion Matrix
- Data Visualization: Tableau Public
- Version Control: GitHub

## Methodology

### Data Cleaning and Exploratory Analysis

The dataset was examined for missing values, duplicate observations, and inconsistencies.

Duplicate records were removed, and sensor measurements were analyzed to understand their distributions and relationships with recorded equipment failures.

Correlation analysis and failure-event profiling were used to investigate sensor behavior.

### Data Splitting and Leakage Prevention

Because the dataset contains repeated observations from individual devices, a random row-level split could introduce data leakage.

Group-based splitting was therefore applied to separate devices across the training, validation, and test datasets.

The validation dataset was used for classification threshold selection, while the test dataset was reserved for final performance evaluation.

### Machine Learning Model

A Random Forest Classifier was trained using nine sensor variables.

Class weighting was applied to account for the imbalance between normal observations and recorded failures.

A Dummy Classifier was also used as a baseline to provide a reference for model evaluation.

### Model Evaluation

The model was evaluated using precision, recall, F1-score, ROC-AUC, average precision, and a confusion matrix.

Accuracy was not treated as the primary performance measure because a classifier predicting normal operation for every observation would achieve very high accuracy while failing to detect equipment failures.

### Classification Threshold Analysis

Multiple classification thresholds were evaluated on the validation dataset.

A threshold of 0.10 was selected and applied to the held-out test dataset.

This approach allows the decision threshold to be assessed separately from the final test evaluation.

## Model Results

The Random Forest model produced the following results on the test dataset at a classification threshold of 0.10:

| Metric | Result |
|---|---:|
| Precision | 1.96% |
| Recall | 16.67% |
| F1-score | 3.51% |
| True Positives | 3 |
| False Positives | 150 |
| True Negatives | 24,736 |
| False Negatives | 15 |

The model correctly identified 3 of the 18 recorded failures in the test dataset.

However, it also generated 150 false positives and failed to identify 15 actual failures.

These findings indicate that the model has limited practical detection performance despite the high overall proportion of correctly classified observations.

The results are presented as an exploratory classification analysis rather than a production-ready maintenance solution.

## Key Findings

- Machine failure events represent a very small proportion of the dataset.
- Classification accuracy is misleading when the target classes are highly imbalanced.
- Reducing the classification threshold increases failure detection sensitivity but also generates more false alarms.
- Sensor feature importance provides insight into the variables contributing to Random Forest predictions.
- The limited number of recorded failures makes reliable model evaluation and threshold selection particularly challenging.
- Additional modeling and validation would be required before considering operational deployment.

## Tableau Dashboards

Two interactive Tableau dashboards were developed to communicate the results.

**Sensor Risk Dashboard**

Presents sensor measurements during normal operation and recorded failures, together with Random Forest feature importance.

**Model Evaluation and Threshold Analysis Dashboard**

Presents the test-set confusion matrix, precision, recall, F1-score, and validation-based classification threshold comparison.

**Tableau Public:** [View Interactive Dashboards](https://public.tableau.com/app/profile/dimitris.votsis/viz/Predictive_Maintenance_Full_Dashboard/PredictiveMaintenanceModelEvaluationThresholdAnalysis)

## Repository Structure

```text
Predictive-Maintenance-ML-Classification/
|
|-- 01_Data/
|   |-- predictive_maintenance.csv
|
|-- 02_Python/
|   |-- maintenance_analysis.py
|
|-- 03_Tableau/
|   |-- Predictive_Maintenance_Full_Dashboard.twbx
|
|-- 04_Screenshots/
|   |-- dashboard_sensor_risk.png
|   |-- dashboard_model_evaluation.png
|   |-- tableau_confusion_matrix.png
|   |-- tableau_validation_threshold.png
|   |-- failure_distribution.png
|   |-- correlation_matrix.png
|   |-- feature_importance.png
|   |-- precision_recall_curve.png
|   |-- confusion_matrix.png
|
|-- 05_Output/
|   |-- failure_event_profiles.csv
|   |-- sensor_importance_ranking.csv
|   |-- model_evaluation_metrics.csv
|   |-- confusion_matrix_results.csv
|   |-- test_predictions.csv
|   |-- threshold_comparison.csv
|
|-- README.md
```

## Limitations and Future Improvements

The current model classifies failures using sensor measurements from the same observation date. It does not forecast failures several days in advance.

Potential improvements include:

- Evaluating alternative classification algorithms.
- Testing additional approaches to class imbalance.
- Engineering historical and time-based sensor features.
- Using repeated device-based validation to assess model stability.
- Incorporating the operational costs of false alarms and missed failures into threshold selection.
- Developing a future-event prediction target for advance maintenance planning.

## Skills Demonstrated

Python, Data Cleaning, Exploratory Data Analysis, Machine Learning, Random Forest, Classification, Imbalanced Data Analysis, Model Evaluation, Data Leakage Prevention, Feature Importance, Tableau, Data Visualization, Business Analysis.
