from sklearn.preprocessing import MinMaxScaler

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'MACD', 'RSI', 'BB_High', 'BB_Low']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Reset index
data = data.reset_index()