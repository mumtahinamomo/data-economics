import os
import csv
import subprocess
import sys

NS = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000, 20000]
SEEDS = [0, 1, 2, 3, 4]

def main():
    here = os.path.dirname(__file__)
    train_py = os.path.join(here, "train.py")
    results_csv = os.path.join(here, "results.csv")

    rows = []
    for n in NS:
        for seed in SEEDS:
            print(f"Doing n={n} seed={seed}")
            cmd = [
                sys.executable, train_py,
                "--n", str(n),
                "--seed", str(seed),
                "--epochs", "10" if n >= 1000 else "7",
                "--max_tokens_per_doc", "200",
                "--max_total_sequences", str(n * 15),
                "--data_path", "/content/data-economics/marginal_value/20000_openwebtext.json",
            ]
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            out = [line for line in out.split("\n") if line.startswith("n=")][-1]
            print(out)

            parts = out.replace("=", " ").split()
            parsed = {parts[i]: parts[i+1] for i in range(0, len(parts), 2)}

            rows.append({
                "n": int(parsed["n"]),
                "seed": int(parsed["seed"]),
                "train_loss": float(parsed["train_loss"]),
                "train_acc": float(parsed["train_acc"]),
                "val_loss": float(parsed["val_loss"]),
                "val_acc": float(parsed["val_acc"]),
            })

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {results_csv}")

if __name__ == "__main__":
    main()