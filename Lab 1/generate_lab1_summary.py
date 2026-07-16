"""
Lab 1 — Summary Document Generator
Pulls together everything built in Lab 1 and writes a self-contained
lab1_summary.md that students bring to Lab 2.

Requires (all in the same folder):
  students_clean.csv          — cleaned dataset (Step 2)
  students_with_entropy.csv   — dataset with attendance_entropy (Step 5)
  reflection.txt              — fairness reflection (Step 6)
"""

import csv
import math
import os
import random
from datetime import date

random.seed(42)
script_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load cleaned data stats
# ─────────────────────────────────────────────────────────────────────────────
clean_path = os.path.join(script_dir, "students_clean.csv")
clean_rows = list(csv.DictReader(open(clean_path)))

n_total     = len(clean_rows)
n_at_risk   = sum(1 for r in clean_rows if r["at_risk"] == "yes")
n_meals     = sum(1 for r in clean_rows if r["meals_program"] == "yes")
att_vals    = [float(r["attendance_rate"]) for r in clean_rows if r["attendance_rate"] != ""]
quiz_vals   = [float(r["avg_quiz_score"])  for r in clean_rows if r["avg_quiz_score"]  != ""]

def stats(vals):
    n  = len(vals)
    mu = sum(vals) / n
    s  = sorted(vals)
    md = s[n // 2]
    return round(mu, 2), round(md, 2), round(min(vals), 2), round(max(vals), 2)

att_mean,  att_med,  att_min,  att_max  = stats(att_vals)
quiz_mean, quiz_med, quiz_min, quiz_max = stats(quiz_vals)

# First 10 rows for the sample table
sample_rows = clean_rows[:10]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Re-run Logistic Regression to get live metrics
# ─────────────────────────────────────────────────────────────────────────────
def col_mu_sd(data, key):
    vals = [r[key] for r in data]
    mu   = sum(vals) / len(vals)
    sd   = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1e-8
    return mu, sd

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))

def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

model_rows = []
for r in clean_rows:
    att  = r["attendance_rate"]
    quiz = r["avg_quiz_score"]
    if att == "" or quiz == "":
        continue
    model_rows.append({
        "attendance_rate": float(att),
        "avg_quiz_score":  float(quiz),
        "meals_program":   1 if r["meals_program"].strip().lower() == "yes" else 0,
        "at_risk":         1 if r["at_risk"].strip().lower() == "yes" else 0,
    })

att_mu,  att_sd  = col_mu_sd(model_rows, "attendance_rate")
quiz_mu, quiz_sd = col_mu_sd(model_rows, "avg_quiz_score")

def feats(row):
    return [(row["attendance_rate"] - att_mu) / att_sd,
            (row["avg_quiz_score"]  - quiz_mu) / quiz_sd,
            row["meals_program"], 1.0]

random.shuffle(model_rows)
split      = int(0.8 * len(model_rows))
train_rows = model_rows[:split]
test_rows  = model_rows[split:]
X_train = [feats(r) for r in train_rows]
y_train = [r["at_risk"] for r in train_rows]
X_test  = [feats(r) for r in test_rows]
y_test  = [r["at_risk"] for r in test_rows]

weights = [0.0] * 4
for _ in range(1000):
    grad = [0.0] * 4
    for x, y in zip(X_train, y_train):
        err = sigmoid(dot(weights, x)) - y
        for j in range(4):
            grad[j] += err * x[j]
    for j in range(4):
        weights[j] -= 0.1 * grad[j] / len(X_train)

y_pred_lr = [1 if sigmoid(dot(weights, x)) >= 0.5 else 0 for x in X_test]

def metrics(y_true, y_pred):
    TP = sum(1 for p, a in zip(y_pred, y_true) if p == 1 and a == 1)
    FP = sum(1 for p, a in zip(y_pred, y_true) if p == 1 and a == 0)
    FN = sum(1 for p, a in zip(y_pred, y_true) if p == 0 and a == 1)
    TN = sum(1 for p, a in zip(y_pred, y_true) if p == 0 and a == 0)
    n  = len(y_true)
    acc = (TP + TN) / n
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return acc, pre, rec, TP, FP, FN, TN

lr_acc, lr_pre, lr_rec, lr_TP, lr_FP, lr_FN, lr_TN = metrics(y_test, y_pred_lr)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Re-run Random Forest to get live metrics
# ─────────────────────────────────────────────────────────────────────────────
def rf_features(row):
    return [row["attendance_rate"], row["avg_quiz_score"], row["meals_program"]]

X_train_rf = [rf_features(r) for r in train_rows]
X_test_rf  = [rf_features(r) for r in test_rows]

def gini(labels):
    n = len(labels)
    if n == 0: return 0.0
    p1 = sum(labels) / n
    return 1.0 - p1 ** 2 - (1 - p1) ** 2

def best_split(X, y, feat_indices):
    best_gain, best_feat, best_thresh = -1, None, None
    parent_g = gini(y)
    n = len(y)
    for fi in feat_indices:
        vals = sorted(set(x[fi] for x in X))
        for i in range(len(vals) - 1):
            thresh  = (vals[i] + vals[i+1]) / 2
            left_y  = [y[k] for k in range(n) if X[k][fi] <= thresh]
            right_y = [y[k] for k in range(n) if X[k][fi] >  thresh]
            if not left_y or not right_y: continue
            gain = parent_g - (len(left_y)/n * gini(left_y) + len(right_y)/n * gini(right_y))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, fi, thresh
    if best_feat is None: return None
    lm = [i for i in range(n) if X[i][best_feat] <= best_thresh]
    rm = [i for i in range(n) if X[i][best_feat] >  best_thresh]
    return best_feat, best_thresh, lm, rm

def build_tree(X, y, feat_indices, max_depth, min_samples):
    if len(y) < min_samples or max_depth == 0 or len(set(y)) == 1:
        return int(round(sum(y) / len(y)))
    result = best_split(X, y, feat_indices)
    if result is None: return int(round(sum(y) / len(y)))
    fi, thresh, li, ri = result
    return {"feature": fi, "threshold": thresh,
            "left":  build_tree([X[i] for i in li], [y[i] for i in li], feat_indices, max_depth-1, min_samples),
            "right": build_tree([X[i] for i in ri], [y[i] for i in ri], feat_indices, max_depth-1, min_samples)}

def predict_tree(node, x):
    if isinstance(node, int): return node
    return predict_tree(node["left"], x) if x[node["feature"]] <= node["threshold"] else predict_tree(node["right"], x)

trees = []
n_tr = len(X_train_rf)
for _ in range(100):
    idx   = [random.randint(0, n_tr - 1) for _ in range(n_tr)]
    X_b   = [X_train_rf[i] for i in idx]
    y_b   = [y_train[i]    for i in idx]
    fi    = random.sample(range(3), 2)
    trees.append(build_tree(X_b, y_b, fi, 6, 5))

y_pred_rf = [1 if sum(predict_tree(t, x) for t in trees) > len(trees)/2 else 0 for x in X_test_rf]
rf_acc, rf_pre, rf_rec, rf_TP, rf_FP, rf_FN, rf_TN = metrics(y_test, y_pred_rf)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load entropy stats
# ─────────────────────────────────────────────────────────────────────────────
entropy_path = os.path.join(script_dir, "students_with_entropy.csv")
ent_rows     = list(csv.DictReader(open(entropy_path)))
ent_vals     = [float(r["attendance_entropy"]) for r in ent_rows]

e_mean = round(sum(ent_vals) / len(ent_vals), 4)
e_med  = round(sorted(ent_vals)[len(ent_vals) // 2], 4)
e_min  = round(min(ent_vals), 4)
e_max  = round(max(ent_vals), 4)

# Top 5 highest entropy students
top5_entropy = sorted(ent_rows, key=lambda r: float(r["attendance_entropy"]), reverse=True)[:5]
# Top 5 lowest (most consistent)
bot5_entropy = sorted(ent_rows, key=lambda r: float(r["attendance_entropy"]))[:5]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Load reflection
# ─────────────────────────────────────────────────────────────────────────────
reflection_path = os.path.join(script_dir, "reflection.txt")
if os.path.exists(reflection_path):
    with open(reflection_path) as f:
        reflection_raw = f.read()
    # Strip the header lines — keep just the student's words
    lines = reflection_raw.strip().splitlines()
    reflection_text = "\n".join(
        l for l in lines if not l.startswith("Lab 1") and not l.startswith("=")
    ).strip()
else:
    reflection_text = "_[No reflection found — run step6_reflection.py first]_"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Write the Markdown summary
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(script_dir, "lab1_summary.md")

md = f"""# Lab 1 Summary — Modeling Equity
**Generated:** {date.today().strftime("%B %d, %Y")}

---

## Dataset Overview

- **Total students:** {n_total}
- **At-risk (yes):** {n_at_risk} ({n_at_risk/n_total*100:.1f}%)
- **On meals program:** {n_meals} ({n_meals/n_total*100:.1f}%)
- **Missing values after imputation:** 0 in `avg_quiz_score`

### Sample Data (first 10 rows)

| student_id | attendance_rate | avg_quiz_score | meals_program | at_risk |
|---|---|---|---|---|
"""

for r in sample_rows:
    md += (f"| {r['student_id']} | {r['attendance_rate']} "
           f"| {r['avg_quiz_score']} | {r['meals_program']} | {r['at_risk']} |\n")

md += f"""
### Column Statistics (after median imputation)

| Column | Mean | Median | Min | Max |
|---|---|---|---|---|
| attendance_rate | {att_mean} | {att_med} | {att_min} | {att_max} |
| avg_quiz_score  | {quiz_mean} | {quiz_med} | {quiz_min} | {quiz_max} |

---

## Step 2 — Median Imputation

Missing values in `avg_quiz_score` were filled using the **median ({quiz_med})**.
The mean shifted by less than 0.02 points, and the min/max were unchanged —
confirming that median imputation keeps the distribution stable.

---

## Step 3 & 4 — Model Results

Both models were trained on 80% of the data ({len(X_train)} rows) and tested
on the remaining 20% ({len(X_test)} rows), using:
- `attendance_rate`
- `avg_quiz_score`
- `meals_program`

### Comparison Table

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | {lr_acc:.1%} | {rf_acc:.1%} |
| **Precision** | {lr_pre:.1%} | {rf_pre:.1%} |
| **Recall** | {lr_rec:.1%} | {rf_rec:.1%} |

### What the metrics mean

- **Accuracy** — Of all students tested, how many were labeled correctly overall.
- **Precision** — Of everyone the tool flagged as at-risk, how many actually were.
- **Recall** — Of every student who truly was at-risk, how many did the tool catch.

### Confusion Matrices

**Logistic Regression**

| | Predicted NO | Predicted YES |
|---|---|---|
| Actually NOT at-risk | TN = {lr_TN} | FP = {lr_FP} |
| Actually AT RISK | FN = {lr_FN} | TP = {lr_TP} |

**Random Forest**

| | Predicted NO | Predicted YES |
|---|---|---|
| Actually NOT at-risk | TN = {rf_TN} | FP = {rf_FP} |
| Actually AT RISK | FN = {rf_FN} | TP = {rf_TP} |

---

## Step 5 — Attendance Entropy

For each student, 6 grading-period attendance rates were simulated and sorted
into 5 buckets (0–20%, 20–40%, 40–60%, 60–80%, 80–100%). Shannon entropy
**H = −Σ pᵢ log(pᵢ)** was calculated from the bucket distribution.

**Low entropy** → consistent attendance.  
**High entropy** → attendance jumps around unpredictably.

### Entropy Summary

| Statistic | Value |
|---|---|
| Mean | {e_mean} |
| Median | {e_med} |
| Min (most consistent) | {e_min} |
| Max (most unpredictable) | {e_max} |
| Max possible (5 equal buckets) | 1.6094 |

### Most Unpredictable Students (top 5)

| student_id | attendance_rate | at_risk | entropy |
|---|---|---|---|
"""
for r in top5_entropy:
    md += f"| {r['student_id']} | {r['attendance_rate']} | {r['at_risk']} | {r['attendance_entropy']} |\n"

md += """
### Most Consistent Students (bottom 5)

| student_id | attendance_rate | at_risk | entropy |
|---|---|---|---|
"""
for r in bot5_entropy:
    md += f"| {r['student_id']} | {r['attendance_rate']} | {r['at_risk']} | {r['attendance_entropy']} |\n"

md += f"""
---

## Step 6 — Fairness Reflection

> {reflection_text.replace(chr(10), chr(10) + '> ')}

---

## Files Produced in Lab 1

| File | Contents |
|---|---|
| `students.csv` | Raw synthetic dataset (300 rows) |
| `students_clean.csv` | Cleaned dataset after median imputation |
| `students_with_entropy.csv` | Full dataset including `attendance_entropy` column |
| `reflection.txt` | Step 6 fairness reflection |
| `lab1_summary.md` | This document |

---

*All data is synthetic — no real student records were used.*
"""

with open(out_path, "w") as f:
    f.write(md)

print("=" * 60)
print("  Lab 1 Summary Document Generated")
print("=" * 60)
print(f"  Saved to: {out_path}")
print()
print(f"  Dataset       : {n_total} students, {n_at_risk} at-risk ({n_at_risk/n_total*100:.1f}%)")
print(f"  LR  metrics   : acc={lr_acc:.1%}  pre={lr_pre:.1%}  rec={lr_rec:.1%}")
print(f"  RF  metrics   : acc={rf_acc:.1%}  pre={rf_pre:.1%}  rec={rf_rec:.1%}")
print(f"  Entropy range : {e_min} – {e_max}  (mean {e_mean})")
if reflection_text.startswith("_[No"):
    print()
    print("  NOTE: No reflection found.")
    print("  Run step6_reflection.py first, then re-run this script.")
print("=" * 60)
