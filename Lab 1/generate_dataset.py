"""
Lab 1 — Modeling Equity
Generate a 300-row synthetic student dataset for a school with limited resources.

Columns
-------
student_id       : unique ID (STU0001 … STU0300)
attendance_rate  : float 0–1, a few values intentionally missing (~3%)
avg_quiz_score   : float 0–100, ~8% missing
meals_program    : yes / no  (free/reduced meals — proxy for low income)
at_risk          : yes / no  (more likely "yes" when attendance & quiz scores are low)
"""

import random
import math
import csv
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ── Helpers ──────────────────────────────────────────────────────────────────

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def normal_sample(mu, sigma, lo=None, hi=None):
    """Box-Muller transform — no external libraries needed."""
    while True:
        u1 = random.random() or 1e-10
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        v = mu + sigma * z
        if lo is not None and v < lo:
            continue
        if hi is not None and v > hi:
            continue
        return v

# ── Generate rows ─────────────────────────────────────────────────────────────
NUM_ROWS = 300
ATTENDANCE_MISSING_RATE = 0.03   # ~3%  → "a few missing values"
QUIZ_MISSING_RATE       = 0.08   # ~8%

rows = []

for i in range(1, NUM_ROWS + 1):
    student_id = f"STU{i:04d}"

    # meals_program — roughly 40% qualify (common in under-resourced schools)
    meals = "yes" if random.random() < 0.40 else "no"

    # attendance_rate — lower on average for meals_program="yes" students
    if meals == "yes":
        att = clamp(round(normal_sample(0.72, 0.15), 4), 0.0, 1.0)
    else:
        att = clamp(round(normal_sample(0.85, 0.12), 4), 0.0, 1.0)

    # avg_quiz_score — correlated with attendance
    quiz_mu = 40 + att * 55        # ranges roughly 40–95 depending on attendance
    quiz = clamp(round(normal_sample(quiz_mu, 12), 1), 0.0, 100.0)

    # at_risk — logistic-style probability based on attendance & quiz score.
    # Calibrated so roughly 20–25% of students are at_risk = "yes", with
    # the probability strongly concentrated among low-attendance/low-score
    # students — matching the research paper's intent.
    score = -4.0 + (1 - att) * 7.0 + (100 - quiz) * 0.06
    prob_at_risk = 1 / (1 + math.exp(-score))
    at_risk = "yes" if random.random() < prob_at_risk else "no"

    # Introduce missing values AFTER computing at_risk (we never want target missing)
    att_val   = None if random.random() < ATTENDANCE_MISSING_RATE  else att
    quiz_val  = None if random.random() < QUIZ_MISSING_RATE        else quiz

    rows.append({
        "student_id":      student_id,
        "attendance_rate": att_val,
        "avg_quiz_score":  quiz_val,
        "meals_program":   meals,
        "at_risk":         at_risk,
    })

# ── Save CSV ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "students.csv")

fieldnames = ["student_id", "attendance_rate", "avg_quiz_score", "meals_program", "at_risk"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ── Print first 10 rows ───────────────────────────────────────────────────────
col_widths = [12, 16, 15, 14, 8]
header = ["student_id", "attendance_rate", "avg_quiz_score", "meals_program", "at_risk"]

def fmt_row(r):
    return (
        f"{r['student_id']:<12}"
        f"{str(r['attendance_rate']):<16}"
        f"{str(r['avg_quiz_score']):<15}"
        f"{r['meals_program']:<14}"
        f"{r['at_risk']:<8}"
    )

print("=" * 65)
print("FIRST 10 ROWS")
print("=" * 65)
print(
    f"{'student_id':<12}"
    f"{'attendance_rate':<16}"
    f"{'avg_quiz_score':<15}"
    f"{'meals_program':<14}"
    f"{'at_risk':<8}"
)
print("-" * 65)
for row in rows[:10]:
    print(fmt_row(row))

# ── Missing value counts ──────────────────────────────────────────────────────
missing = {col: sum(1 for r in rows if r[col] is None) for col in fieldnames}

print("\n" + "=" * 65)
print("MISSING VALUE COUNTS (out of 300 rows)")
print("=" * 65)
for col, count in missing.items():
    pct = count / NUM_ROWS * 100
    bar = "█" * count
    print(f"  {col:<20} {count:>3} missing  ({pct:.1f}%)")

print(f"\nCSV saved to: {csv_path}")
