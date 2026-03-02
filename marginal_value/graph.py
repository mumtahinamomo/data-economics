import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

grouped = df.groupby("n").mean().reset_index()

plt.plot(grouped["n"], grouped["val_acc"], marker="o")

plt.xlabel("Number of Documents")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Doc Size")

plt.grid(True)
plt.savefig("graphplot.png")
plt.show()