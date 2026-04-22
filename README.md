# Employee Churn Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

Predicts which employees are likely to leave using a Random Forest model trained on real HR data. Achieves **92% accuracy** and **89% recall** on the minority class after SMOTE balancing.

---

## Overview

Employee turnover is expensive. This project builds an end-to-end classification pipeline that identifies at-risk employees before they leave — giving HR teams time to act.

The pipeline covers data preparation, class balancing with SMOTE, model training, evaluation, and an interactive dashboard for real-time prediction.

---

## Features

| Feature | Description |
|---|---|
| ETL Pipeline | Automated cleaning, encoding, and feature engineering with Pandas |
| Class Balancing | SMOTE to address the 5:1 class imbalance in the dataset |
| ML Model | Random Forest Classifier with `class_weight='balanced'` |
| Evaluation | Accuracy, Precision, Recall and F1-score per class |
| Dashboard | Interactive Streamlit app for real-time employee risk prediction |

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 92% |
| Precision (churn) | 95% |
| Recall (churn) | 89% |
| F1-score (churn) | 92% |

> Recall is the primary metric here. Missing an employee who is about to leave is more costly than a false alarm.

---

## Tech Stack

- **Language:** Python 3.12
- **Data:** Pandas
- **ML:** Scikit-learn, imbalanced-learn (SMOTE)
- **Dashboard:** Streamlit
- **Architecture:** Modular, function-based design

---

## Project Structure

```
churn-classifier/
├── src/
│   ├── prepare.py       # ETL pipeline — cleaning, encoding, feature engineering
│   └── train.py         # Model training and evaluation
├── app.py               # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/ghcalado/churn-classifier.git
cd churn-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Download the IBM HR Analytics dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place `Wa_Fn-UseC_-HR-Employee-Attrition.csv` in the root folder.

**4. Run the pipeline**
```bash
python src/train.py
```

**5. Launch the dashboard**
```bash
streamlit run app.py
```

---

## How It Works

**Data preparation** — removes constant columns (`EmployeeCount`, `StandardHours`, `Over18`), encodes binary columns with `map()`, and applies One Hot Encoding to categorical features via `get_dummies()`.

**Class balancing** — the dataset has 1,233 employees who stayed and only 237 who left. SMOTE generates synthetic samples of the minority class, bringing both to 1,233 before training.

**Model** — Random Forest with `class_weight='balanced'` as an additional safeguard. Trained on 80% of the data, evaluated on the remaining 20%.

**Dashboard** — Streamlit interface where you can adjust employee attributes via sliders and get an instant prediction with probability score and feature importance chart.

---

## Roadmap

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] XGBoost comparison
- [ ] FastAPI endpoint for production serving
- [ ] SHAP values for explainability

---

## Author

**Ghabriel Calado**  
Computer Science Student | Python & Data  
[GitHub](https://github.com/ghcalado) · [LinkedIn](https://www.linkedin.com/in/ghabriel-calado-7132a33b6/)
