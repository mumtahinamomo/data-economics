import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/data-economics/marginal_value/results.csv")

grouped = df.groupby("n")["val_acc"].mean().reset_index()
grouped = grouped.sort_values("n")

plt.figure(figsize=(10, 6))
plt.plot(grouped["n"], grouped["val_acc"], marker="o")
plt.xlabel("Number of Documents")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Doc Size")
plt.grid(True)
plt.savefig("/content/data-economics/marginal_value/graphplot.png")
plt.show()
print("saved :))")