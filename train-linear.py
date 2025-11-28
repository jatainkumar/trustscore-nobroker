import json
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Generate Mock Data (Same as before)
X = []
y = []
for _ in range(1000):
    streak = np.random.randint(0, 24)
    delay = np.random.randint(-5, 30)
    utility = np.random.rand()
    linkedin = np.random.choice([0, 1])
    
    # Simple linear logic
    true_score = 500 + (streak * 10) - (delay * 12) + (utility * 100) + (linkedin * 50)
    X.append([streak, delay, utility, linkedin])
    y.append(max(300, min(900, int(true_score + np.random.randint(-30, 30)))))

# 2. Train
model = LinearRegression()
model.fit(X, y)

# 3. Export
data = {
    "type": "linear",
    "intercept": model.intercept_,
    "coefficients": list(model.coef_)
}

with open("model_linear.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved model_linear.json")