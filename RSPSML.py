# © 2025 azank1 / StratDev. All Rights Reserved.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file):
    df = pd.read_csv(file, parse_dates=["time"], index_col="time")
    df = df[['close']]
    return df

def feature_engineering(df):
    df['return'] = df['close'].pct_change()
    df['rolling_10'] = df['return'].rolling(window=10).mean()
    df['rolling_30'] = df['return'].rolling(window=30).mean()
    df['rolling_60'] = df['return'].rolling(window=60).mean()
    df['roc'] = df['close'].pct_change(periods=10)  # Rate of Change over 10 periods
    
    df['trend_state'] = np.where(df['rolling_10'] > df['rolling_30'], 1, 
                          np.where(df['rolling_10'] < df['rolling_30'], -1, 0))  # 1 = Uptrend, 0 = Neutral, -1 = Downtrend
    
    df.dropna(inplace=True)
    return df

file = 'BTC.csv'  # Change to TOTAL.csv if needed
df = load_data(file)
df = feature_engineering(df)

features = ['return', 'rolling_10', 'rolling_30', 'rolling_60', 'roc']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df['trend_state'].values  # Target: Trend states (-1, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class TrendPredictionModel(nn.Module):
    def __init__(self):
        super(TrendPredictionModel, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = TrendPredictionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

y_pred = model(X_test_torch).detach().numpy()
y_pred = np.where(y_pred > 0.7, 1, np.where(y_pred < 0.3, -1, 0))  # Thresholds for trend classification

df_test = df.iloc[len(df) - len(y_test):]
df_test['prediction'] = y_pred

plt.figure(figsize=(14, 6))
plt.plot(df_test.index, df_test['close'], label='Price', color='black')
plt.scatter(df_test.index[df_test['prediction'] == 1], df_test['close'][df_test['prediction'] == 1], color='green', label='Enter Long')
plt.scatter(df_test.index[df_test['prediction'] == -1], df_test['close'][df_test['prediction'] == -1], color='red', label='Exit to Cash')
plt.scatter(df_test.index[df_test['prediction'] == 0], df_test['close'][df_test['prediction'] == 0], color='yellow', label='Hold')
plt.legend()
plt.title("Trend Prediction Model")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
s
