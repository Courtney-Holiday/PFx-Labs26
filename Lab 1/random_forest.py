"""
Lab 1 — Step 4: Random Forest
Reads students_clean.csv, uses the same 80/20 split and features as the
Logistic Regression script, trains a Random Forest built from scratch,
then prints a side-by-side comparison table.

No external libraries required — built entirely with Python stdlib.
"""

import csv
import math
import os
import random

random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "students_clean.csv")

rows = []
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        att  = row["attendance_rate"]
        quiz = row["avg_quiz_score"]
        if att == "" or quiz == "":
            continue
        rows.append({
            "attendance_rate": float(att),
            "avg_quiz_score":  float(quiz),
            "meals_program":   1 if row["meals_program"].strip().lower() == "yes" else 0,
            "at_risk":         1 if row["at_risk"].strip().lower() == "yes" else 0,
        })

# ── Feature extraction (no scaling needed for trees) ─────────────────────────
def features(row):
    return [
        row["attendance_rate"],
        row["avg_quiz_score"],
        row["meals_program"],
    ]

# ── Same 80/20 split as logistic regression (same seed, same shuffle) ─────────
random.shuffle(rows)
split      = int(0.8 * len(rows))
train_rows = rows[:split]
test_rows  = rows[split:]

X_train = [features(r) for r in train_rows]
y_train = [r["at_risk"]  for r in train_rows]
X_test  = [features(r) for r in test_rows]
y_test  = [r["at_risk"]  for r in test_rows]

# ─────────────────────────────────────────────────────────────────────────────
# Decision Tree (CART-style, binary splits, Gini impurity)
# ─────────────────────────────────────────────────────────────────────────────

def gini(labels):
    """Gini impurity of a label list."""
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = sum(labels) / n
    return 1.0 - p1 ** 2 - (1 - p1) ** 2

def best_split(X, y, feature_indices):
    """
    Find the best (feature, threshold) split among the given feature_indices.
    Returns (feature_idx, threshold, left_mask, right_mask) or None if no
    improvement is possible.
    """
    best_gain  = -1
    best_feat  = None
    best_thresh = None
    parent_gini = gini(y)
    n = len(y)

    for fi in feature_indices:
        values = sorted(set(x[fi] for x in X))
        # Try midpoints between consecutive unique values
        thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values) - 1)]
        for thresh in thresholds:
            left_y  = [y[i] for i in range(n) if X[i][fi] <= thresh]
            right_y = [y[i] for i in range(n) if X[i][fi] >  thresh]
            if not left_y or not right_y:
                continue
            gain = parent_gini - (
                len(left_y)  / n * gini(left_y) +
                len(right_y) / n * gini(right_y)
            )
            if gain > best_gain:
                best_gain   = gain
                best_feat   = fi
                best_thresh = thresh

    if best_feat is None:
        return None
    left_mask  = [i for i in range(n) if X[i][best_feat] <= best_thresh]
    right_mask = [i for i in range(n) if X[i][best_feat] >  best_thresh]
    return best_feat, best_thresh, left_mask, right_mask

def build_tree(X, y, feature_indices, max_depth, min_samples):
    """
    Recursively build a decision tree.
    Returns a dict representing a node, or a leaf value (int 0/1).
    """
    # Leaf conditions
    if len(y) < min_samples or max_depth == 0 or len(set(y)) == 1:
        return int(round(sum(y) / len(y)))  # majority vote

    result = best_split(X, y, feature_indices)
    if result is None:
        return int(round(sum(y) / len(y)))

    fi, thresh, left_idx, right_idx = result
    X_left  = [X[i] for i in left_idx];  y_left  = [y[i] for i in left_idx]
    X_right = [X[i] for i in right_idx]; y_right = [y[i] for i in right_idx]

    return {
        "feature":   fi,
        "threshold": thresh,
        "left":  build_tree(X_left,  y_left,  feature_indices, max_depth - 1, min_samples),
        "right": build_tree(X_right, y_right, feature_indices, max_depth - 1, min_samples),
    }

def predict_tree(node, x):
    """Traverse a tree node to get a prediction for sample x."""
    if isinstance(node, int):
        return node
    if x[node["feature"]] <= node["threshold"]:
        return predict_tree(node["left"],  x)
    else:
        return predict_tree(node["right"], x)

# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

N_TREES       = 100   # number of trees in the forest
MAX_DEPTH     = 6     # max depth per tree
MIN_SAMPLES   = 5     # min samples to keep splitting
N_FEATURES    = 2     # features randomly sampled per tree (sqrt of 3 ≈ 2)

trees = []
n_train = len(X_train)

for t in range(N_TREES):
    # Bootstrap sample (sample with replacement)
    indices    = [random.randint(0, n_train - 1) for _ in range(n_train)]
    X_boot = [X_train[i] for i in indices]
    y_boot = [y_train[i] for i in indices]

    # Random feature subset
    feat_indices = random.sample(range(len(X_train[0])), N_FEATURES)

    tree = build_tree(X_boot, y_boot, feat_indices, MAX_DEPTH, MIN_SAMPLES)
    trees.append(tree)

def predict_forest(trees, x):
    """Majority vote across all trees."""
    votes = [predict_tree(t, x) for t in trees]
    return 1 if sum(votes) > len(votes) / 2 else 0

y_pred_rf = [predict_forest(trees, x) for x in X_test]

# ── Metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    TP = sum(1 for p, a in zip(y_pred, y_true) if p == 1 and a == 1)
    FP = sum(1 for p, a in zip(y_pred, y_true) if p == 1 and a == 0)
    FN = sum(1 for p, a in zip(y_pred, y_true) if p == 0 and a == 1)
    TN = sum(1 for p, a in zip(y_pred, y_true) if p == 0 and a == 0)
    n  = len(y_true)
    accuracy  = (TP + TN) / n
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return accuracy, precision, recall, TP, FP, FN, TN

rf_acc, rf_pre, rf_rec, TP, FP, FN, TN = compute_metrics(y_test, y_pred_rf)

# Logistic Regression results from Step 3 (reproduced for the comparison table)
lr_acc, lr_pre, lr_rec = 0.881, 0.818, 0.643

# ── Print comparison table ────────────────────────────────────────────────────
print("=" * 62)
print("  Lab 1 — Model Comparison: Logistic Regression vs. Random Forest")
print("=" * 62)
print(f"  {'Metric':<14} {'Logistic Regression':>22} {'Random Forest':>16}")
print("-" * 62)
print(f"  {'Accuracy':<14} {lr_acc:>22.1%} {rf_acc:>16.1%}")
print(f"  {'Precision':<14} {lr_pre:>22.1%} {rf_pre:>16.1%}")
print(f"  {'Recall':<14} {lr_rec:>22.1%} {rf_rec:>16.1%}")
print("=" * 62)
print()
print("  Random Forest — Confusion Matrix (test set)")
print(f"  {'':28} Predicted NO   Predicted YES")
print(f"  {'Actually NOT at-risk':<28} TN={TN:<6}        FP={FP}")
print(f"  {'Actually AT RISK':<28} FN={FN:<6}        TP={TP}")
print("=" * 62)
print()
print("  Random Forest settings:")
print(f"    Trees         : {N_TREES}")
print(f"    Max depth     : {MAX_DEPTH}")
print(f"    Features/tree : {N_FEATURES} (random subset of 3 total)")
print(f"    Bootstrap     : yes (sample with replacement)")
