# Lab 1 Summary — Modeling Equity
**Generated:** July 16, 2026

---

## Dataset Overview

- **Total students:** 300
- **At-risk (yes):** 80 (26.7%)
- **On meals program:** 120 (40.0%)
- **Missing values after imputation:** 0 in `avg_quiz_score`

### Sample Data (first 10 rows)

| student_id | attendance_rate | avg_quiz_score | meals_program | at_risk |
|---|---|---|---|---|
| STU0001 | 0.799 | 82.2 | no | no |
| STU0002 | 0.9123 | 100.0 | no | no |
| STU0003 | 0.7763 | 95.8 | yes | no |
| STU0004 | 0.9979 | 100.0 | no | no |
| STU0005 | 0.8298 | 91.9 | no | no |
| STU0006 | 0.8912 | 100.0 | no | no |
| STU0007 | 0.7332 | 78.1 | no | no |
| STU0008 | 0.7817 | 99.7 | yes | no |
| STU0009 | 0.8612 | 74.8 | no | no |
| STU0010 | 0.7478 | 95.2 | no | no |

### Column Statistics (after median imputation)

| Column | Mean | Median | Min | Max |
|---|---|---|---|---|
| attendance_rate | 0.8 | 0.8 | 0.39 | 1.0 |
| avg_quiz_score  | 82.64 | 82.7 | 53.9 | 100.0 |

---

## Step 2 — Median Imputation

Missing values in `avg_quiz_score` were filled using the **median (82.7)**.
The mean shifted by less than 0.02 points, and the min/max were unchanged —
confirming that median imputation keeps the distribution stable.

---

## Step 3 & 4 — Model Results

Both models were trained on 80% of the data (232 rows) and tested
on the remaining 20% (59 rows), using:
- `attendance_rate`
- `avg_quiz_score`
- `meals_program`

### Comparison Table

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | 88.1% | 81.4% |
| **Precision** | 81.8% | 100.0% |
| **Recall** | 64.3% | 21.4% |

### What the metrics mean

- **Accuracy** — Of all students tested, how many were labeled correctly overall.
- **Precision** — Of everyone the tool flagged as at-risk, how many actually were.
- **Recall** — Of every student who truly was at-risk, how many did the tool catch.

### Confusion Matrices

**Logistic Regression**

| | Predicted NO | Predicted YES |
|---|---|---|
| Actually NOT at-risk | TN = 43 | FP = 2 |
| Actually AT RISK | FN = 5 | TP = 9 |

**Random Forest**

| | Predicted NO | Predicted YES |
|---|---|---|
| Actually NOT at-risk | TN = 45 | FP = 0 |
| Actually AT RISK | FN = 11 | TP = 3 |

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
| Mean | 0.5522 |
| Median | 0.6365 |
| Min (most consistent) | 0.0 |
| Max (most unpredictable) | 1.3297 |
| Max possible (5 equal buckets) | 1.6094 |

### Most Unpredictable Students (top 5)

| student_id | attendance_rate | at_risk | entropy |
|---|---|---|---|
| STU0250 | 0.7089 | no | 1.3297 |
| STU0115 | 0.6414 | yes | 1.2425 |
| STU0181 | 0.6166 | no | 1.2425 |
| STU0204 | 0.3894 | yes | 1.2425 |
| STU0055 | 0.6763 | no | 1.0986 |

### Most Consistent Students (bottom 5)

| student_id | attendance_rate | at_risk | entropy |
|---|---|---|---|
| STU0004 | 0.9979 | no | 0.0 |
| STU0006 | 0.8912 | no | 0.0 |
| STU0013 | 0.9832 | no | 0.0 |
| STU0017 | 0.8242 | no | 0.0 |
| STU0020 | 0.9498 | no | 0.0 |

---

## Step 6 — Fairness Reflection

> Using meals_program as a feature means the tool is partly predicting
> risk based on a family's income level rather than just the student's
> own behavior or learning. This could cause the model to consistently
> flag low-income students as at-risk even when their actual attendance
> and quiz scores are fine, simply because the pattern in the training
> data connected income to risk. If a school used this tool to decide
> who gets extra support, students from wealthier families might be
> overlooked even when they are struggling, while low-income students
> might be labeled as problems before they have had a chance to show
> what they can do. A fairer approach would be to rely only on
> attendance and academic performance directly, and involve teachers or
> counselors in any final decision rather than letting the algorithm
> decide alone.

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
