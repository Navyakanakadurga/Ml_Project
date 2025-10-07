import streamlit as st
import pandas as pd, numpy as np, pickle
from pathlib import Path
st.set_page_config(layout='wide', page_title='Expense Forecast & Budget Optimizer')
st.title('Personal Expense Forecasting & Budget Optimizer — Enhanced')

data_path = Path('data/processed/cleaned_transactions.csv')
if not data_path.exists():
    st.error('Processed data not found. Run preprocessing first.')
else:
    df = pd.read_csv(data_path, parse_dates=['date'])
    st.sidebar.header('Settings')
    months_to_show = st.sidebar.slider('Months to show', 6, 36, 18)
    forecast_horizon = st.sidebar.slider('Forecast horizon (months)', 1, 6, 1)
    st.header('Transactions sample')
    st.dataframe(df.head(10))
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    monthly = df[df['is_expense']].groupby('year_month').agg(total_spent=('amount_abs','sum')).reset_index()
    monthly = monthly.sort_values('year_month')
    st.subheader('Monthly total expenses')
    st.line_chart(monthly.set_index('year_month')['total_spent'].tail(months_to_show))
    # Per-category trend selector
    st.subheader('Category breakdown')
    cat = st.selectbox('Select category', ['all'] + sorted(df['category'].unique().tolist()))
    if cat != 'all':
        mc = df[(df['is_expense']) & (df['category']==cat)].groupby('year_month').agg(total_spent=('amount_abs','sum')).reset_index()
        st.line_chart(mc.set_index('year_month')['total_spent'].tail(months_to_show))
    else:
        st.write('Showing all categories aggregated above.')
    # Load models
    model_path = Path('models/rf_monthly_total.pkl')
    lstm_path = Path('models/lstm_monthly_total.h5')
    if model_path.exists():
        with open(model_path,'rb') as f:
            rf = pickle.load(f)
    else:
        rf = None
    if lstm_path.exists():
        try:
            from tensorflow.keras.models import load_model
            lstm = load_model(str(lstm_path))
        except Exception as e:
            lstm = None
    else:
        lstm = None
    # Prepare features for prediction
    last_months = monthly.tail(4)['total_spent'].tolist()
    st.subheader('Forecasts (next month)')
    if len(last_months) >= 4:
        if rf:
            feat = np.array([last_months[-1], last_months[-2], last_months[-3], last_months[-4]]).reshape(1,-1)
            pred_rf = rf.predict(feat)[0]
            st.metric('RandomForest prediction (next month)', f'{pred_rf:,.2f}')
        if lstm is not None:
            seq = np.array(last_months[-12:]) if len(last_months) >= 12 else np.array([0]* (12 - len(last_months)) + last_months)
            seq = seq.reshape((1, len(seq), 1))
            pred_lstm = lstm.predict(seq)[0][0]
            st.metric('LSTM prediction (next month)', f'{pred_lstm:,.2f}')
        else:
            st.info('LSTM model not available in models/.')
    else:
        st.warning('Not enough monthly data to produce forecasts.')
    # Simple budget optimizer
    st.header('Budget optimizer (simple)')
    monthly_avg = monthly['total_spent'].tail(6).mean() if len(monthly)>=6 else monthly['total_spent'].mean()
    user_goal = st.number_input('Set desired monthly spending goal', min_value=100.0, value=float(monthly_avg))
    st.write(f'Current recent average (last 6 months): {monthly_avg:,.2f}')
    if user_goal < monthly_avg:
        reduction = monthly_avg - user_goal
        pct = (reduction / monthly_avg) * 100
        st.success(f'To reach your goal, reduce spending by {reduction:,.2f} ({pct:.1f}%). Suggested per-category caps:')
        # naive suggestion: cap categories proportionally by recent share
        cat_share = df[df['is_expense']].groupby('category').agg(total=('amount_abs','sum')).reset_index()
        cat_share['share'] = cat_share['total'] / cat_share['total'].sum()
        cat_share['suggested_cap'] = cat_share['share'] * user_goal
        st.dataframe(cat_share.sort_values('share', ascending=False).head(20))
    else:
        st.info('Your goal is higher than or equal to recent average — no reduction needed.')
