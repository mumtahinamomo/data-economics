import json
from datasets import load_dataset

ds = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True
)

out_path = "openwebtext_100.json"
n = 500

data = []
for doc in ds.take(n):
    data.append({"text": doc["text"]})

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)