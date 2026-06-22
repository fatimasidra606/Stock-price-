import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Open": [100, 102, 104, 103, 105, 107, 110, 108],
    "High": [102, 105, 106, 105, 108, 110, 112, 111],
    "Low": [99, 101, 102, 101, 103, 105, 108, 107],
    "Volume": [2000, 2200, 2100, 2300, 2400, 2500, 2600, 2550],
    "Close": [101, 104, 105, 104, 107, 109, 111, 110]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Stock Prices")
plt.show()

new_data = np.array([[109, 112, 107, 2700]])
predicted_price = model.predict(new_data)

print("\nPredicted Closing Price:", predicted_price[0])
