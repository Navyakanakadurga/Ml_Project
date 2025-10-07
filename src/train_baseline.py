import argparse, pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
def main(input_csv, output_model):
    df = pd.read_csv(input_csv)
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
    monthly = df[df['is_expense']].groupby('year_month').agg(total_spent=('amount_abs','sum')).reset_index()
    monthly = monthly.sort_values('year_month')
    monthly['total_spent_lag1'] = monthly['total_spent'].shift(1)
    monthly['total_spent_lag2'] = monthly['total_spent'].shift(2)
    monthly['total_spent_lag3'] = monthly['total_spent'].shift(3)
    monthly = monthly.dropna().reset_index(drop=True)
    monthly['target_next'] = monthly['total_spent'].shift(-1)
    monthly = monthly.dropna().reset_index(drop=True)
    X = monthly[['total_spent','total_spent_lag1','total_spent_lag2','total_spent_lag3']].values
    y = monthly['target_next'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X,y)
    with open(output_model,'wb') as f:
        pickle.dump(model,f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.input, args.output)
