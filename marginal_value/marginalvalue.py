import json
from datasets import load_dataset

ds = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True
)

out_path = "openwebtext.json"

for n in [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000]:
    ds.shuffle()

    data = []
    for doc in ds.take(n):
        data.append({"text": doc["text"]})


    with open("" + str(n) + "_" + out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)