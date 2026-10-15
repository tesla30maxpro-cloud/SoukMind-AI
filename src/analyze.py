
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data/predictions.csv")
df["date"] = pd.to_datetime(df["date"])

# Residual = Actual - Predicted
df["residual"] = df["units_sold"] - df["prediction"]

# Plot 1: Actual vs Predicted
plt.figure(figsize=(11, 5))

plt.plot(df["date"], df["units_sold"],
         marker="o", label="Actual")

plt.plot(df["date"], df["prediction"],
         marker="x", label="Predicted")

plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.title("SoukMind - Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Residuals vs Predictions
plt.figure(figsize=(9, 5))

plt.scatter(df["prediction"], df["residual"])
plt.axhline(0, linestyle="--", color="red")

plt.xlabel("Predicted Sales")
plt.ylabel("Residual")
plt.title("SoukMind - Residual Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nResidual Statistics:")
print(df["residual"].describe())
