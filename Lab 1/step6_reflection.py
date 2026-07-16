"""
Lab 1 — Step 6: Fairness Reflection
Prompts the student to write 3-5 sentences about the fairness of using
meals_program to predict at_risk, then saves their answer.
"""

import os

print("=" * 68)
print("  Lab 1 — Step 6: Think It Through")
print("=" * 68)
print()
print("  Is it fair to use meals_program (which tells us about a family's")
print("  income) to predict which students need help?")
print()
print("  What could go wrong if a school used this exact tool to decide")
print("  who gets extra support?")
print()
print("  Write 3-5 sentences below. Press ENTER twice when you're done.")
print("-" * 68)

lines = []
while True:
    line = input()
    if line == "" and lines and lines[-1] == "":
        break
    lines.append(line)

# Remove trailing blank lines
while lines and lines[-1] == "":
    lines.pop()

reflection = "\n".join(lines).strip()

if not reflection:
    print("\n  No response entered. Please re-run and type your answer.")
else:
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    output_path   = os.path.join(script_dir, "reflection.txt")
    with open(output_path, "w") as f:
        f.write("Lab 1 — Step 6 Fairness Reflection\n")
        f.write("=" * 68 + "\n\n")
        f.write(reflection + "\n")
    print()
    print("  Saved to reflection.txt")
    print(f"  Word count: {len(reflection.split())}")
