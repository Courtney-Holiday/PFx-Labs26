"""
Lab 1 — Step 3: Logistic Regression
Reads students_clean.csv, splits 80/20 train/test, trains a logistic regression
model using gradient descent, then reports accuracy, precision, and recall.

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
        # Skip rows where attendance_rate is still missing
        if att == "" or quiz == "":
            continue
        rows.append({
            "attendance_rate": float(att),
            "avg_quiz_score":  float(quiz),
            "meals_program":   1 if row["meals_program"].strip().lower() == "yes" else 0,
            "at_risk":         1 if row["at_risk"].strip().lower() == "yes" else 0,
        })

# ── Feature scaling (standardize) ────────────────────────────────────────────
# Logistic regression converges faster when features are on a similar scale.

def col_stats(data, key):
    vals = [r[key] for r in data]
    mu   = sum(vals) / len(vals)
    sd   = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1e-8
    return mu, sd

att_mu,  att_sd  = col_stats(rows, "attendance_rate")
quiz_mu, quiz_sd = col_stats(rows, "avg_quiz_score")
# meals_program is already 0/1 — no scaling needed

def features(row):
    return [
        (row["attendance_rate"] - att_mu)  / att_sd,
        (row["avg_quiz_score"]  - quiz_mu) / quiz_sd,
        row["meals_program"],
        1.0,   # bias term
    ]

# ── Train / test split (80 / 20, shuffled) ───────────────────────────────────
random.shuffle(rows)
split      = int(0.8 * len(rows))
train_rows = rows[:split]
test_rows  = rows[split:]

X_train = [features(r) for r in train_rows]
y_train = [r["at_risk"]  for r in train_rows]
X_test  = [features(r) for r in test_rows]
y_test  = [r["at_risk"]  for r in test_rows]

# ── Logistic regression via gradient descent ──────────────────────────────────

def sigmoid(z):
    # Clamp to avoid overflow
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def predict_prob(w, x):
    return sigmoid(dot(w, x))

# Hyperparameters
LEARNING_RATE = 0.1
EPOCHS        = 1000

n_features = len(X_train[0])
weights    = [0.0] * n_features

for epoch in range(EPOCHS):
    # Compute gradients
    grad = [0.0] * n_features
    for x, y in zip(X_train, y_train):
        error = predict_prob(weights, x) - y
        for j in range(n_features):
            grad[j] += error * x[j]
    # Update weights
    for j in range(n_features):
        weights[j] -= LEARNING_RATE * grad[j] / len(X_train)

# ── Predict on test set (threshold = 0.5) ────────────────────────────────────
def predict(w, x, threshold=0.5):
    return 1 if predict_prob(w, x) >= threshold else 0

# With the calibrated dataset (~27% at-risk), the default 0.5 threshold works well.
THRESHOLD = 0.5
y_pred = [predict(weights, x, threshold=THRESHOLD) for x in X_test]

# ── Metrics ───────────────────────────────────────────────────────────────────
#
#  Terminology (positive class = at_risk = 1):
#    TP  — predicted at_risk AND actually at_risk
#    FP  — predicted at_risk BUT not actually at_risk
#    FN  — predicted NOT at_risk BUT actually at_risk
#    TN  — predicted NOT at_risk AND not actually at_risk

TP = sum(1 for p, a in zip(y_pred, y_test) if p == 1 and a == 1)
FP = sum(1 for p, a in zip(y_pred, y_test) if p == 1 and a == 0)
FN = sum(1 for p, a in zip(y_pred, y_test) if p == 0 and a == 1)
TN = sum(1 for p, a in zip(y_pred, y_test) if p == 0 and a == 0)

n_test   = len(y_test)
accuracy  = (TP + TN) / n_test
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

# ── Display results ───────────────────────────────────────────────────────────
print("=" * 62)
print("  Lab 1 — Logistic Regression Results")
print("=" * 62)
print(f"  Training rows  : {len(X_train)}")
print(f"  Test rows      : {len(X_test)}")
print(f"  Decision threshold: {THRESHOLD}")
print()
print(f"  {'Metric':<12} {'Score':>8}   Plain-English Meaning")
print("-" * 62)
print(f"  {'Accuracy':<12} {accuracy:>8.1%}   Of ALL students in the test set,")
print(f"  {'':12} {'':8}   how many did we label correctly?")
print()
print(f"  {'Precision':<12} {precision:>8.1%}   Of everyone the tool FLAGGED as")
print(f"  {'':12} {'':8}   at-risk, how many actually were?")
print()
print(f"  {'Recall':<12} {recall:>8.1%}   Of every student who IS actually")
print(f"  {'':12} {'':8}   at-risk, how many did we catch?")
print("=" * 62)
print()
print("  Confusion matrix (test set)")
print(f"  {'':25} Predicted NO   Predicted YES")
print(f"  {'Actually NOT at-risk':<25} TN={TN:<6}        FP={FP}")
print(f"  {'Actually AT RISK':<25} FN={FN:<6}        TP={TP}")
print("=" * 62)
