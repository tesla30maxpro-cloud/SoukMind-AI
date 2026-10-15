
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "data").mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n = 180

# Generate synthetic retail data
dates = pd.date_range("2025-01-01", periods=n)

price = rng.uniform(20, 70, n)
ads = rng.uniform(0, 120, n)
discount = rng.choice([0, 0.1, 0.2, 0.3], n)
weekend = (dates.dayofweek >= 5).astype(int)
noise = rng.normal(0, 5, n)

sales = (
    100
    - 1.0 * price
    + 0.15 * ads
    + 50 * discount
    + 8 * weekend
    + noise
)

sales = np.maximum(0, np.rint(sales)).astype(int)

df = pd.DataFrame({
    "date": dates,
    "price": price,
    "ad_budget": ads,
    "discount_pct": discount,
    "weekend": weekend,
    "units_sold": sales
})

df.to_csv(ROOT / "data" / "sales.csv", index=False)

# Chronological train/test split
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

features = [
    "price", "ad_budget", "discount_pct", "weekend"
]

X_train = train[features]
y_train = train["units_sold"]

X_test = test[features]
y_test = test["units_sold"]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)





ROOT = Path(__file__).resolve().parent.parent

# Ensure the directory exists
(ROOT / "data").mkdir(parents=True, exist_ok=True)

# Copy test data
results = test[["date", "units_sold"]].copy()

# Add predicted values
results["prediction"] = predictions

# Residuals = Actual - Prediction
results["residual"] = (
    results["units_sold"] - results["prediction"]
)

# Save to CSV
output_path = ROOT / "data" / "predictions.csv"

results.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(results.head())

# Baseline: always predict training mean
baseline = np.full(len(y_test), y_train.mean())

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")

evaluate("Baseline", y_test, baseline)
evaluate("Linear Regression", y_test, predictions)


print("\nIntercept:", model.intercept_)
print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")
