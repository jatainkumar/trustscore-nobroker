import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import _tree

# 1. Generate Complex Data (Non-linear patterns)
X = []
y = []
for _ in range(1000):
    streak = np.random.randint(0, 24)
    delay = np.random.randint(-5, 30)
    utility = np.random.rand()
    linkedin = np.random.choice([0, 1])
    
    # Complex logic: Penalty for delay is worse if you have no streak
    penalty_multiplier = 1.5 if streak < 3 else 1.0
    
    true_score = 500 + (streak * 10) - (delay * 12 * penalty_multiplier) + (utility * 100) + (linkedin * 50)
    X.append([streak, delay, utility, linkedin])
    y.append(max(300, min(900, int(true_score))))

# 2. Train (Limit depth/estimators for JSON size)
model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
model.fit(X, y)

# 3. Helper to export Tree to Dictionary
def tree_to_dict(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            return {
                "feature_index": int(tree_.feature[node]), # 0=streak, 1=delay, etc.
                "threshold": float(threshold),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            return {"value": float(tree_.value[node][0][0])}

    return recurse(0)

# 4. Export all trees in the forest
forest_data = {
    "type": "forest",
    "n_estimators": len(model.estimators_),
    "trees": []
}

for estimator in model.estimators_:
    forest_data["trees"].append(tree_to_dict(estimator, ["streak", "delay", "utility", "linkedin"]))

with open("model_rf.json", "w") as f:
    json.dump(forest_data, f, indent=2)
print("Saved model_rf.json")
