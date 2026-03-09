import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

BASE_DIR     = "/Users/shamsimumtahinamomo/Documents/111A- Spring 2026/Data economics/inequality_and_marketpower"
MARGINAL_DIR = "/Users/shamsimumtahinamomo/Documents/111A- Spring 2026/Data economics/marginal_value"

bengali_csv = os.path.join(BASE_DIR, "results_bengali.csv")
english_csv = os.path.join(MARGINAL_DIR, "results.csv")
output_png  = os.path.join(BASE_DIR, "graph_bengali_vs_english.png")

bn = pd.read_csv(bengali_csv)
en = pd.read_csv(english_csv)

bn_grouped = bn.groupby("n").agg(
    mean_acc=("val_acc", "mean"),
    std_acc=("val_acc",  "std")
).reset_index()

en_grouped = en.groupby("n").agg(
    mean_acc=("val_acc", "mean"),
    std_acc=("val_acc",  "std")
).reset_index()

bn_grouped["n_pct"] = bn_grouped["n"] / bn_grouped["n"].max() * 100
en_grouped["n_pct"] = en_grouped["n"] / en_grouped["n"].max() * 100

Y_MIN, Y_MAX = 0.15, 0.45

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bengali plot
axes[0].plot(bn_grouped["n"], bn_grouped["mean_acc"], marker="o",
             color="#E8514A", linewidth=2)
axes[0].fill_between(bn_grouped["n"],
    bn_grouped["mean_acc"] - bn_grouped["std_acc"],
    bn_grouped["mean_acc"] + bn_grouped["std_acc"],
    alpha=0.2, color="#E8514A")
axes[0].set_title("Bengali Wikipedia\n(Low-Resource)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Number of Documents")
axes[0].set_ylabel("Top-25 Accuracy")
axes[0].set_ylim(Y_MIN, Y_MAX)
axes[0].grid(True, alpha=0.3)

# English plot
axes[1].plot(en_grouped["n"], en_grouped["mean_acc"], marker="o",
             color="#4A90D9", linewidth=2)
axes[1].fill_between(en_grouped["n"],
    en_grouped["mean_acc"] - en_grouped["std_acc"],
    en_grouped["mean_acc"] + en_grouped["std_acc"],
    alpha=0.2, color="#4A90D9")
axes[1].set_title("English OpenWebText\n(High-Resource)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Number of Documents")
axes[1].set_ylabel("Top-25 Accuracy")
axes[1].set_ylim(Y_MIN, Y_MAX)
axes[1].grid(True, alpha=0.3)

# Comparison plot
axes[2].plot(bn_grouped["n_pct"], bn_grouped["mean_acc"], marker="o",
             color="#E8514A", linewidth=2, label="Bengali (max=500 docs)")
axes[2].fill_between(bn_grouped["n_pct"],
    bn_grouped["mean_acc"] - bn_grouped["std_acc"],
    bn_grouped["mean_acc"] + bn_grouped["std_acc"],
    alpha=0.2, color="#E8514A")
axes[2].plot(en_grouped["n_pct"], en_grouped["mean_acc"], marker="o",
             color="#4A90D9", linewidth=2, label="English (max=500 docs)")
axes[2].fill_between(en_grouped["n_pct"],
    en_grouped["mean_acc"] - en_grouped["std_acc"],
    en_grouped["mean_acc"] + en_grouped["std_acc"],
    alpha=0.2, color="#4A90D9")

bn_ceiling = bn_grouped["mean_acc"].max()
en_floor   = en_grouped["mean_acc"].min()
axes[2].axhline(y=bn_ceiling, color="#E8514A", linestyle="--", alpha=0.6,
                label=f"Bengali ceiling ({bn_ceiling:.2f})")
axes[2].axhline(y=en_floor,   color="#4A90D9", linestyle="--", alpha=0.6,
                label=f"English floor ({en_floor:.2f})")

axes[2].set_title(" Comparison\n(% of Available Corpus Used)", fontsize=13, fontweight="bold")
axes[2].set_xlabel("% of Corpus Used")
axes[2].set_ylabel("Top-25 Accuracy")
axes[2].set_ylim(Y_MIN, Y_MAX)
axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.suptitle("Data Inequality: Marginal Value of Training Data\nBengali (Low-Resource) vs English (High-Resource)",
             fontsize=14, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {output_png}")