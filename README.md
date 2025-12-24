# Fraud Detection for E-Commerce and Bank Transactions

## Overview
This project develops machine learning models to detect fraud in e-commerce and bank credit transactions at Adey Innovations Inc. It addresses class imbalance, feature engineering (e.g., geolocation, time-based), and model explainability using SHAP.

## Project Structure
- `data/raw/`: Original datasets (add to .gitignore).
- `data/processed/`: Cleaned data (Parquet format).
- `notebooks/`: EDA, engineering, modeling Jupyter notebooks.
- `src/`: Reusable Python modules (e.g., preprocessing).
- `tests/`: Unit tests.
- `models/`: Saved ML models.
- `scripts/`: Automation scripts.

## Setup
1. Clone repo: `git clone <repo-url>`.
2. Install deps: `pip install -r requirements.txt`.
3. Download datasets to `data/raw/`:
   - [Fraud_Data.csv](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce-dataset) (or similar).
   - [IpAddress_to_Country.csv](from Kaggle or MaxMind sample).
   - [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
4. Run notebooks: `jupyter notebook notebooks/`.

## Interim-1 Progress
- Data cleaning, EDA, feature engineering complete for both datasets.
- Class imbalance handled with SMOTE strategy.
- See `notebooks/` for details.

## Interim-2: Model Building
- Run `notebooks/modeling.ipynb` after EDA/feature-eng for baselines/ensembles.
- Metrics: AUC-PR ~0.8+ for ensemble; see saved models in /models/.
- Best Model: XGBoost (justified in notebook).
- Setup: Same as Interim-1; run `pytest tests/` for validation.