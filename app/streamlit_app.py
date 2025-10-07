
import streamlit as st
import pandas as pd, pickle, numpy as np
from pathlib import Path
st.title('Personal Expense Forecasting — Demo App')
st.markdown('This demo uses a saved RandomForest baseline model to predict next-month total expenses.')
data_path = Path('data/processed/cleaned_transactions.csv')
if not data_path.exists():
    st.error('Processed data not found. Run preprocessing first.')
else:
    df = pd.read_csv(data_path)
    st.write('Sample transactions:')
    st.dataframe(df.head(10))
    # aggregate monthly
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    monthly = df[df['is_expense']].groupby('year_month').agg(total_spent=('amount_abs','sum')).reset_index()
    st.line_chart(monthly.set_index('year_month')['total_spent'])
    # load model
    model_path = Path('models/rf_monthly_total.pkl')
    if model_path.exists():
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        # prepare last available lags
        last = monthly.sort_values('year_month').tail(3)['total_spent'].tolist()
        if len(last) < 3:
            st.warning('Not enough months for lag features.')
        else:
            cur_total = monthly.sort_values('year_month').tail(1)['total_spent'].values[0]
            features = np.array([cur_total, last[-1], last[-2], last[-3]]).reshape(1,-1)
            pred = model.predict(features)[0]
            st.metric('Predicted next-month total expense', f'{pred:,.2f}')
    else:
        st.warning('Model artifact not found in models/')
