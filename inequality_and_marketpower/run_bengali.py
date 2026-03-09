import os
import csv
import subprocess
import sys

BASE_DIR = "/Users/shamsimumtahinamomo/Documents/111A- Spring 2026/Data economics/inequality_and_marketpower"

NS = [50, 100, 200, 300, 400, 500]
SEEDS = [0, 1, 2]


def main():
    train_py = os.path.join(BASE_DIR, "train_bengali.py")
    results_csv = os.path.join(BASE_DIR, "results_bengali.csv")

    rows = []
    for n in NS:
        for seed in SEEDS:
            print(f"Running n={n} seed={seed}...")
            cmd = [
                sys.executable, train_py,
                "--n", str(n),
                "--seed", str(seed),
                "--epochs", "3",
                "--max_tokens_per_doc", "200",
                "--max_total_sequences", "10000",
            ]
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                print(out)

                parts = out.replace("=", " ").split()
                parsed = {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}

                rows.append({
                    "n":        int(parsed["n"]),
                    "seed":     int(parsed["seed"]),
                    "val_loss": float(parsed["val_loss"]),
                    "val_acc":  float(parsed["val_acc"]),
                })
            except Exception as e:
                print(f"  Skipped n={n} seed={seed}: {e}")

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {results_csv}")


if __name__ == "__main__":
    main()