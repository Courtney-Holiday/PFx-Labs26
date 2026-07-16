"""
Lab 1 — Step 5: Attendance Entropy
For each student, simulate 6 grading-period attendance rates, sort each into
one of 5 buckets, then compute Shannon entropy H = -Σ pi * log(pi).

Buckets:
  0 = under 20%    (0.00 – 0.19)
  1 = 20–40%       (0.20 – 0.39)
  2 = 40–60%       (0.40 – 0.59)
  3 = 60–80%       (0.60 – 0.79)
  4 = 80–100%      (0.80 – 1.00)

Entropy interpretation:
  Low  → student's attendance is consistent (mostly one bucket)
  High → student's attendance jumps around unpredictably across buckets

Output: students_with_entropy.csv  (all original columns + attendance_entropy)
"""

import csv
import math
import os
import random

random.seed(42)

GRADING_PERIODS = 6
N_BUCKETS       = 5

def bucket(att_rate):
    """Assign an attendance rate to one of 5 buckets (0-indexed)."""
    if att_rate < 0.20:
        return 0
    elif att_rate < 0.40:
        return 1
    elif att_rate < 0.60:
        return 2
    elif att_rate < 0.80:
        return 3
    else:
        return 4

def shannon_entropy(counts):
    """
    H = -Σ pi * log(pi)  (natural log, base e — consistent with information theory).
    counts is a list of raw counts per bucket.
    Returns 0.0 for a perfectly consistent student (only one bucket used).
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)   # natural log; use math.log2(p) for bits
    return h

def simulate_period_attendances(base_rate, n_periods, noise_sd=0.12):
    """
    Generate n_periods attendance rates for a student whose overall rate is
    base_rate.  A small random walk adds realistic variation across periods.
    Values are clamped to [0, 1].
    """
    rates = []
    for _ in range(n_periods):
        # Box-Muller single sample
        import random as _r
        u1 = _r.random() or 1e-10
        u2 = _r.random()
        z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        val = base_rate + noise_sd * z
        rates.append(max(0.0, min(1.0, val)))
    return rates

# ── Load cleaned data ─────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
input_path  = os.path.join(script_dir, "students_clean.csv")
output_path = os.path.join(script_dir, "students_with_entropy.csv")

rows = []
with open(input_path, newline="") as f:
    reader     = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

# ── Compute entropy for each student ─────────────────────────────────────────
for row in rows:
    raw_att = row["attendance_rate"]

    # If attendance_rate is still missing, use 0.5 as a neutral base
    base = float(raw_att) if raw_att != "" else 0.5

    # Simulate 6 grading-period rates
    period_rates = simulate_period_attendances(base, GRADING_PERIODS)

    # Count how many periods fall into each bucket
    counts = [0] * N_BUCKETS
    for r in period_rates:
        counts[bucket(r)] += 1

    # Shannon entropy
    h = shannon_entropy(counts)
    row["attendance_entropy"] = round(h, 4)

    # Store period rates for transparency (optional — not written to CSV)
    row["_periods"] = [round(r, 3) for r in period_rates]

# ── Save updated CSV ──────────────────────────────────────────────────────────
out_fields = fieldnames + ["attendance_entropy"]
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

# ── Print first 10 rows ───────────────────────────────────────────────────────
print("=" * 72)
print("  Lab 1 — attendance_entropy  (first 10 students)")
print("=" * 72)
print(f"  {'student_id':<12} {'att_rate':<12} {'at_risk':<10} "
      f"{'periods (6)':<32} {'entropy'}")
print("-" * 72)
for row in rows[:10]:
    att_disp = row["attendance_rate"] if row["attendance_rate"] != "" else "N/A"
    print(
        f"  {row['student_id']:<12}"
        f"{str(att_disp):<12}"
        f"{row['at_risk']:<10}"
        f"{str(row['_periods']):<32}"
        f"{row['attendance_entropy']}"
    )

# ── Summary statistics on entropy column ─────────────────────────────────────
entropies = [row["attendance_entropy"] for row in rows]
n         = len(entropies)
mean_h    = sum(entropies) / n
sorted_h  = sorted(entropies)
median_h  = sorted_h[n // 2]
max_h     = max(entropies)
min_h     = min(entropies)

# Max possible entropy for 6 periods across 5 buckets
max_possible = math.log(min(GRADING_PERIODS, N_BUCKETS))

print()
print("=" * 72)
print("  Entropy Summary (all 300 students)")
print("=" * 72)
print(f"  Mean entropy    : {mean_h:.4f}")
print(f"  Median entropy  : {median_h:.4f}")
print(f"  Min entropy     : {min_h:.4f}  (most consistent attendance)")
print(f"  Max entropy     : {max_h:.4f}  (most unpredictable attendance)")
print(f"  Max possible H  : {max_possible:.4f}  "
      f"(all {min(GRADING_PERIODS,N_BUCKETS)} buckets equally used)")
print()
print(f"  Saved to: {output_path}")
