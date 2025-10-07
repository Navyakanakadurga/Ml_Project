import pandas as pd, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
def create_sequences(values, seq_len=12):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)
def main(input_csv='data/processed/cleaned_transactions.csv', seq_len=12):
    df = pd.read_csv(input_csv, parse_dates=['date'])
    monthly = df[df['is_expense']].groupby(df['date'].dt.to_period('M').astype(str)).agg(total_spent=('amount_abs','sum')).reset_index()
    values = monthly['total_spent'].values.astype(float)
    if len(values) <= seq_len:
        print('Not enough data to train LSTM. Need more than', seq_len, 'months.')
        return
    X, y = create_sequences(values, seq_len=seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    split = int(len(X)*0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    model = Sequential([
        LSTM(64, input_shape=(seq_len,1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ckpt = ModelCheckpoint('models/lstm_monthly_total.h5', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=200, batch_size=8, callbacks=[es, ckpt])
    model.save('models/lstm_monthly_total.h5')
if __name__ == '__main__':
    main()
