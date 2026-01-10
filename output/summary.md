## Executive Summary

The logistic regression model predicts heart disease with about 87% accuracy and reasonably calibrated probabilities (Brier score 0.113). The most influential factor families are exercise-related response, ST segment changes, chest pain type, thallium categories, and core clinical measures like cholesterol, blood pressure, age, and sex.

---

## Key Findings

- A relatively simple logistic regression model can reliably predict heart disease status in this dataset, making it practical for screening or decision-support.
- Exercise-related measures (Max HR, Exercise angina) and ST segment characteristics (ST depression, Slope of ST_2) are central to distinguishing higher vs. lower heart disease risk.
- Encoded chest pain patterns (Chest pain type_4) and thallium imaging categories (Thallium_7) carry substantial predictive signal, suggesting that detailed diagnostic categories add value beyond basic vitals.
- Traditional risk factors—Cholesterol, BP, Age, and Sex—remain important contributors, reinforcing their role in risk assessment alongside exercise-test findings.
- The dataset covers a wide clinical range (e.g., Cholesterol from 126 to 564, BP from 94 to 200), so the model has learned from both low- and high-risk profiles, but the sample size (270 patients) limits generalizability.

---

## Model Performance

- **Best model:** Logistic Regression  
  - Accuracy 0.87: correctly classifies about 87 out of 100 patients.  
  - Recall 0.87: identifies about 87% of patients with heart disease.  
  - ROC AUC 0.91: strong ability to separate patients with and without heart disease.  
  - Cross-validation accuracy ~0.85 ± 0.06: performance is fairly stable across different splits.  
  - Brier score 0.1132: indicates **reasonably calibrated probabilities**, suitable for risk scoring.

- **Random Forest (comparison model):**  
  - Accuracy 0.81, ROC AUC 0.89, CV accuracy ~0.79 ± 0.06.  
  - Brier score 0.1313: slightly worse calibration than logistic regression, meaning its risk probabilities are a bit less reliable.

- Overall, logistic regression is both more accurate and better calibrated than the random forest, making it the preferred model for both prediction and risk estimation.

---

## Top Predictive Factors

Based on **Random Forest feature importance**:

- **Max HR** – Most influential feature; how high heart rate rises during testing is strongly linked to predicted heart disease risk.
- **ST depression** – Degree of ST depression is a major signal of risk.
- **Chest pain type_4** – This specific chest pain category is highly informative for distinguishing risk levels.
- **Thallium_7** – This thallium imaging category contributes strongly to the model’s decisions.
- **Cholesterol** – Higher or lower cholesterol levels meaningfully affect predicted risk.
- **Age** – Older age increases predicted risk.
- **BP** – Resting blood pressure is an important contributor.
- **Exercise angina** – Presence or absence of angina during exercise is a key risk indicator.
- **Slope of ST_2** – This particular ST slope category adds additional nuance to risk assessment.
- **Sex** – Biological sex has a measurable, though smaller, impact on predictions.

---

## Recommendations

- **Use logistic regression as the primary risk model** for this dataset, especially when calibrated probability estimates (risk scores) are needed for clinical or operational decisions.
- **Prioritize collection and quality of exercise-test and ST segment data** (Max HR, ST depression, Exercise angina, Slope of ST_2), as these are among the strongest predictors.
- **Incorporate detailed diagnostic categories** (Chest pain type_4, Thallium_7) into assessment workflows where available, since they add predictive value beyond basic vitals.
- **Validate the model on a larger and more diverse population** before deployment, and consider periodic recalibration to maintain probability accuracy over time.