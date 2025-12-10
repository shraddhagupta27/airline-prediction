# airline-prediction

# Deciphering Airline Performance
Built a machine learning pipeline using 4M+ U.S. flight records to predict pre-departure delays and enable proactive operational alerts.

## Overview
This project develops an end-to-end machine learning workflow to predict whether a scheduled commercial flight will be delayed before departure. Using large-scale U.S. Bureau of Transportation Statistics (BTS) data, the pipeline includes data preparation, exploratory data analysis (EDA), feature engineering, class balancing, hyperparameter tuning, and generating a tuned XGBoost model capable of identifying high-risk flights ahead of time.

## Problem Statement
Airlines and airports face costly operational disruptions caused by flight delays. Identifying delay-prone flights before departure allows ground crews, dispatch, and customer service teams to mitigate downstream issues such as gate changes, crew conflicts, and missed connections. This project focuses on predicting pre-departure delays using historical flight, schedule, and weather-related data to support more proactive decision-making.

## Dataset

Source: U.S. Bureau of Transportation Statistics (BTS)
Data volume: 4,000,000+ rows
Includes: Flight schedule data, Carrier identifiers, Origin & destination airports, Distance, airtime, cancellations, diversion flags

## Tools & Technologies
- Python: pandas, numpy, matplotlib, seaborn
- Machine Learning: scikit-learn, XGBoost, Random Forest
- Optimization: Optuna (hyperparameter tuning)
- Data Processing: class balancing, missing-value handling, encoding

## Methods
- Data ingestion & cleaning (handling missing values, filtering out corrupted rows)
- Exploratory Data Analysis (EDA) to identify delay patterns & correlations
- Feature engineering: Time-based features (hour, day, month, season), Historical delay tendencies
- Class imbalance handling using under-sampling / SMOTE
- Model training using Decision Tree, Random Forest, and XGBoost
- Hyperparameter tuning with Optuna to optimize recall/F1 on minority class
- Model evaluation using Accuracy, Recall, Precision, F1-score, Confusion Matrix

## Key Insights

- Delay occurrences vary heavily by time of day, with evening flights showing higher risk.
- After optimization, XGBoost significantly outperformed traditional models.
- Final model achieved ~62% Recall and ~50% F1-Score on the delay class—substantial uplift vs. baseline.






