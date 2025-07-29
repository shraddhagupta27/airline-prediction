# airline-prediction

# Flight Delay Prediction Before Departure (4M+ Records)

## Project Overview
This project focuses on building a machine learning solution to predict flight delays before departure using over 4 million historical flight records from the U.S. Bureau of Transportation Statistics (BTS). The primary goal is to help airlines and airports anticipate delays proactively, minimizing disruptions and improving passenger communication.


## Key Highlights

- Objective: Predict whether a flight will be delayed (15+ mins) before departure, using scheduled metadata.
- Data Prep: Extensive data cleaning, null handling, time-based feature engineering (hour, weekday, route, etc.)
- Imbalance Handling: Class imbalance (71% no-delay) addressed via `class_weight='balanced'` and `BalancedRandomForestClassifier`
- Modeling:
  - Benchmarked 4+ models: Decision Tree, Random Forest, Balanced RF, XGBoost
  - Hyperparameter tuning via Optuna
  - Final model: Tuned XGBoost → Recall: 62% | F1: 50% | PR-AUC: 0.48
- Evaluation Metrics: Chosen to emphasize F1 and PR-AUC, critical for real-world alerting systems


## Why This Matters
Delays cost airlines billions each year and frustrate passengers. This model helps surface high-risk flights before takeoff, giving operations teams time to act — from resource allocation to proactive alerts.

##  Key Learnings
- How to design ML systems under operational constraints (no real-time weather/ATC input)
- Using **Optuna** for efficient hyperparameter search over complex model spaces


