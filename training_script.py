import json
import random
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Generate Mock Data
# We create 1000 synthetic tenant profiles to train our model
print("Generating synthetic data...")
X = []
y = []

for _ in range(1000):
    # Features
    streak = random.randint(0, 24)       # Months of continuous payment
    delay = random.randint(-5, 30)       # Average delay days (-5 is early)
    utility_score = random.random()      # 0.0 to 1.0 (Utility bill consistency)
    linkedin = random.choice([0, 1])     # 1 if LinkedIn verified
    
    # Logic for Ground Truth (The "Real" Score)
    # Base 500
    # +10 points per streak month
    # -10 points per day of delay (capped penalty)
    # +100 points for perfect utility history
    # +50 points for LinkedIn
    
    true_score = 500 + (streak * 10) - (delay * 12) + (utility_score * 100) + (linkedin * 50)
    
    # Add some random noise (to make it realistic for ML)
    noise = random.randint(-30, 30)
    final_score = max(300, min(900, int(true_score + noise)))
    
    X.append([streak, delay, utility_score, linkedin])
    y.append(final_score)

# 2. Train the Model
print("Training Linear Regression Model...")
X = np.array(X)
y = np.array(y)

model = LinearRegression()
model.fit(X, y)

print(f"Model R^2 Score: {model.score(X, y):.4f}")

# 3. Export Model Weights
# We save the intercept and coefficients so the JavaScript app can use them
model_data = {
    "metadata": {
        "algorithm": "Linear Regression",
        "trained_on": "1000 synthetic records",
        "features": ["streak", "delay", "utility", "linkedin"]
    },
    "weights": {
        "intercept": model.intercept_,
        "coefficients": list(model.coef_)
    }
}

filename = "model.json"
with open(filename, "w") as f:
    json.dump(model_data, f, indent=4)

print(f"Success! Model saved to {filename}")
print("Upload this 'model.json' to your GitHub repository along with the HTML files.")
