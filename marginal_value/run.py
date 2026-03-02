import os
import csv
import subprocess
import sys

NS = [50, 100, 200, 300, 400, 500]
SEEDS = [0, 1, 2]

def main():
    here = os.path.dirname(__file__)
    train_py = os.path.join(here, "train.py")
    results_csv = os.path.join(here, "results.csv")

    rows = []
    for n in NS:
        for seed in SEEDS:
            cmd = [
                 sys.executable, train_py,
                 "--n", str(n),
                 "--seed", str(seed),
                 "--epochs", "1",
                 "--max_tokens_per_doc", "200",
                 "--max_total_sequences", "20000",
]
            out = subprocess.check_output(cmd, text=True).strip()
            print(out)

            parts = out.replace("=", " ").split()
            parsed = {parts[i]: parts[i+1] for i in range(0, len(parts), 2)}

            rows.append({
                "n": int(parsed["n"]),
                "seed": int(parsed["seed"]),
                "val_loss": float(parsed["val_loss"]),
                "val_acc": float(parsed["val_acc"]),
            })

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {results_csv}")

if __name__ == "__main__":
    main()