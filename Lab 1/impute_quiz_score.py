"""
Lab 1 — Step 2: Median Imputation
Reads students.csv, fills missing avg_quiz_score values with the column median,
prints a before/after stats comparison, and saves the result as students_clean.csv.
"""

import csv
import os
import math

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_stats(values):
    """Return (mean, median, min, max) for a list of floats. Ignores None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None, None, None
    n = len(nums)
    mean   = sum(nums) / n
    sorted_nums = sorted(nums)
    mid    = n // 2
    median = sorted_nums[mid] if n % 2 == 1 else (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    return mean, median, min(nums), max(nums)

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
input_path  = os.path.join(script_dir, "students.csv")
output_path = os.path.join(script_dir, "students_clean.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
rows = []
with open(input_path, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        # Convert numeric columns; keep empty strings as None
        row["attendance_rate"] = (
            float(row["attendance_rate"]) if row["attendance_rate"] != "" else None
        )
        row["avg_quiz_score"] = (
            float(row["avg_quiz_score"]) if row["avg_quiz_score"] != "" else None
        )
        rows.append(row)

# ── Before stats ──────────────────────────────────────────────────────────────
quiz_before = [r["avg_quiz_score"] for r in rows]
missing_before = sum(1 for v in quiz_before if v is None)
mean_b, median_b, min_b, max_b = compute_stats(quiz_before)

# ── Compute median (excluding missing) and impute ─────────────────────────────
impute_value = median_b   # already computed from non-null values above

for row in rows:
    if row["avg_quiz_score"] is None:
        row["avg_quiz_score"] = impute_value

# ── After stats ───────────────────────────────────────────────────────────────
quiz_after = [r["avg_quiz_score"] for r in rows]
missing_after = sum(1 for v in quiz_after if v is None)
mean_a, median_a, min_a, max_a = compute_stats(quiz_after)

# ── Print comparison ──────────────────────────────────────────────────────────
print("=" * 58)
print("  avg_quiz_score — Before vs. After Median Imputation")
print("=" * 58)
print(f"  {'Statistic':<18} {'BEFORE':>12} {'AFTER':>12}")
print("-" * 58)
print(f"  {'Missing values':<18} {missing_before:>12} {missing_after:>12}")
print(f"  {'Mean':<18} {mean_b:>12.2f} {mean_a:>12.2f}")
print(f"  {'Median':<18} {median_b:>12.2f} {median_a:>12.2f}")
print(f"  {'Minimum':<18} {min_b:>12.2f} {min_a:>12.2f}")
print(f"  {'Maximum':<18} {max_b:>12.2f} {max_a:>12.2f}")
print("=" * 58)
print(f"\n  Imputed value used: {impute_value:.2f} (the median of non-missing scores)")
print(f"  Rows filled in    : {missing_before}")

# ── Save cleaned CSV ──────────────────────────────────────────────────────────
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        # Write numeric values rounded nicely; attendance may still have Nones
        out = dict(row)
        out["attendance_rate"] = (
            "" if out["attendance_rate"] is None else round(out["attendance_rate"], 4)
        )
        out["avg_quiz_score"] = (
            "" if out["avg_quiz_score"] is None else round(out["avg_quiz_score"], 1)
        )
        writer.writerow(out)

print(f"\n  Cleaned data saved to: {output_path}")
