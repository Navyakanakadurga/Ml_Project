# Personal Expense Forecasting and Budget Optimization

This repository contains a professional, reproducible starter project for *Personal Expense Forecasting and Budget Optimization*.
It was generated from the user's provided project description and datasets.
## Structure
- `data/raw/` - original uploaded datasets
- `data/processed/` - cleaned and preprocessed data (cleaned_transactions.csv)
- `notebooks/` - suggested notebooks for EDA and modelling (sample files included)
- `src/` - processing and modeling scripts
- `models/` - trained model artifacts (RandomForest baseline)
- `app/` - Streamlit app to visualize and get forecasts
- `reports/` - evaluation reports and artifacts

## Quick start (local)
1. Create a Python virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
2. Run preprocessing:
```bash
python src/data_preprocessing.py --input data/raw --output data/processed/cleaned_transactions.csv
```
3. Train baseline model:
```bash
python src/train_baseline.py --input data/processed/cleaned_transactions.csv --output models/rf_monthly_total.pkl
```
4. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## What I included
- Basic cleaning & normalization of uploaded CSVs
- Monthly aggregation and simple lag-feature supervised dataset
- RandomForest baseline model predicting next-month total expenses
- Saved model and JSON report with MAE/RMSE across time-series splits
- Starter Streamlit app and scripts for reproducibility
- Dockerfile and requirements.txt for deployment

## Next steps (recommended)
- Expand feature engineering (holiday flags, external economic indicators)
- Train category-level models and deep-learning (LSTM/Transformer)
- Add unit tests and CI pipeline
- Add more robust categorical mapping and merchant NLP cleaning
- Create richer Streamlit UI for per-category forecasts and budget optimization
