import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("BTC.csv", parse_dates=["time"], index_col="time")

df["log_close"] = np.log(df["close"])
df["return"] = df["log_close"].diff()
df["ma_short"] = df["return"].rolling(10).mean()
df["ma_long"] = df["return"].rolling(30).mean()
df.dropna(inplace=True)

X = df[["return", "ma_short", "ma_long"]]
y = np.where(df["ma_short"] > df["ma_long"], 1, -1)  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression()
clf.fit(X_scaled, y)

print("Coefficients:", clf.coef_)      
print("Intercept:", clf.intercept_)    
print("Scaler mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)
