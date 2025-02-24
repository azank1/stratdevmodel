import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Example dataset with columns: time, close
df = pd.read_csv("BTC.csv", parse_dates=["time"], index_col="time")

# Simple features: log returns and rolling means
df["log_close"] = np.log(df["close"])
df["return"] = df["log_close"].diff()
df["ma_short"] = df["return"].rolling(10).mean()
df["ma_long"] = df["return"].rolling(30).mean()
df.dropna(inplace=True)

# Define X, y
X = df[["return", "ma_short", "ma_long"]]
y = np.where(df["ma_short"] > df["ma_long"], 1, -1)  # Simplify to 2-class: +1 or -1

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_scaled, y)

# Print model parameters
print("Coefficients:", clf.coef_)      # shape (1, 3) for 3 features
print("Intercept:", clf.intercept_)    # shape (1,)
print("Scaler mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)
