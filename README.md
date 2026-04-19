# 🛠️ Industrial Predictive Maintenance: Sensor Risk Analysis

## 📌 Project Overview
This project focuses on **Predictive Maintenance** for industrial equipment using **Machine Learning**. By analyzing real-world sensor data (Temperature, Rotational Speed, Torque, etc.), the goal is to identify patterns that lead to equipment failure and build a predictive model to prevent costly downtime.

## 🎯 Objectives
*   Analyze high-frequency sensor data to detect anomalies.
*   Determine the **Root Causes** of machine breakdowns using statistical profiling.
*   Implement a **Random Forest Classifier** to predict failures before they occur.
*   Develop an interactive **Tableau Dashboard** for real-time risk monitoring.

## 🛠 Tools & Technologies
*   **Language:** Python (Pandas, Matplotlib, Seaborn)
*   **Machine Learning:** Scikit-learn (Random Forest)
*   **BI Visualization:** Tableau Desktop / Tableau Public
*   **Analytics:** Feature Importance & Correlation Heatmaps

## 📊 Methodology & Process
1.  **Data Pre-processing:** Cleaned sensor metrics and handled categorical machine data.
2.  **Exploratory Data Analysis (EDA):** Performed correlation analysis to see which sensors (Metrics 1-9) react during failure windows.
3.  **Machine Learning:** Trained a Random Forest model using **Stratified Splitting** to handle the imbalanced nature of failure events.
4.  **Insight Generation:** Identified the **Top Failure Drivers** through Feature Importance ranking.

## 📈 Key Results
*   **Critical Thresholds:** Identified specific sensor levels (e.g., Temperature and Torque) that consistently correlate with 95% of failure events.
*   **Predictive Power:** The model achieved high accuracy in distinguishing normal operations from high-risk states.
*   **Proactive Strategy:** Ranked the most critical sensors to prioritize maintenance efforts.

## 📷 Dashboard & Visuals
### Industrial Monitoring Dashboard
![Predictive Maintenance Dashboard](../04_Screenshots/Predictive_Maintenance_Full_Dashboard.png)

## 🔗 Live Interactive Dashboard
[View on Tableau Public](https://public.tableau.com/views/Predictive_Maintenance_Full_Dashboard/PredictiveMaintenanceDashboardEnergySector?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 🚀 Skills Demonstrated
*   **Predictive Analytics** & Classification
*   **Industrial IoT Data Handling**
*   **Machine Learning Interpretability** (Feature Importance)
*   **Data Storytelling** (Tableau Design)

---