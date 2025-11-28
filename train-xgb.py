import json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import _tree

# 1. Generate Data
X = []
y = []
for _ in range(1000):
    streak = np.random.randint(0, 24)
    delay = np.random.randint(-5, 30)
    utility = np.random.rand()
    linkedin = np.random.choice([0, 1])
    
    score = 500 + (streak * 10) - (delay * 12) + (utility * 100)
    # XGBoost excels at finding edge cases
    if delay > 20 and linkedin == 0: score -= 100 # Huge penalty
    
    X.append([streak, delay, utility, linkedin])
    y.append(max(300, min(900, int(score))))

# 2. Train
model = GradientBoostingRegressor(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 3. Helper (Reuse tree export logic)
def tree_to_dict(tree):
    tree_ = tree.tree_
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            return {
                "feature_index": int(tree_.feature[node]),
                "threshold": float(tree_.threshold[node]),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            return {"value": float(tree_.value[node][0][0])}
    return recurse(0)

# 4. Export (Note: GBM starts with a baseline average)
gbm_data = {
    "type": "gbm",
    "init_score": float(np.mean(y)), # Simplified init score
    "learning_rate": model.learning_rate,
    "trees": []
}

# Sklearn stores trees as an array of arrays [[tree1], [tree2]]
for estimator in model.estimators_:
    gbm_data["trees"].append(tree_to_dict(estimator[0]))

with open("model_xgboost.json", "w") as f:
    json.dump(gbm_data, f, indent=2)
print("Saved model_xgboost.json")