import os
import argparse
import json
import random
import re
import numpy as np
import tensorflow.keras as keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer


# Cleaning 
def clean_text(doc: str) -> str:
    doc = doc.replace("--", " ")
    doc = doc.lower()
    doc = re.sub(r"[^a-z\s]+", " ", doc)
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc


def load_docs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
    return texts


def build_sequences_from_docs(
    docs,
    tokenizer: Tokenizer,
    seq_length: int,
    max_tokens_per_doc: int,
    max_total_sequences: int,
):
    sequences = []
    total = 0

    for d in docs:
        if not d:
            continue

        ids = tokenizer.texts_to_sequences([d])[0]
        if not ids:
            continue

        if max_tokens_per_doc is not None and len(ids) > max_tokens_per_doc:
            ids = ids[:max_tokens_per_doc]

        if len(ids) <= seq_length:
            continue

        for i in range(seq_length, len(ids)):
            seq = ids[i - seq_length : i + 1]  
            sequences.append(seq)
            total += 1
            if max_total_sequences is not None and total >= max_total_sequences:
                return np.array(sequences, dtype=np.int32)

    return np.array(sequences, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    default_data_path = os.path.join(os.path.dirname(__file__), "openwebtext_100.json")
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--n", type=int, default=None, help="Number of documents to sample (with replacement).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--max_tokens_per_doc", type=int, default=400, help="Cap tokens per doc for speed.")
    parser.add_argument("--max_total_sequences", type=int, default=200000, help="Cap total sequences for speed.")
    args = parser.parse_args()

    if args.n is None:
        args.n = int(input("Enter n (number of documents to sample): ").strip())

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    raw_docs = load_docs(args.data_path)
    if not raw_docs:
        raise ValueError(f"No documents loaded from: {args.data_path}")

    cleaned_docs = [clean_text(t) for t in raw_docs if isinstance(t, str) and t.strip()]
    m = len(cleaned_docs)
    if m < 2:
        raise ValueError("Need at least 2 documents total.")

    holdout = max(1, int(0.1 * m))
    val_docs = cleaned_docs[:holdout]
    pool = cleaned_docs[holdout:]
    sampled_indices = random.sample(range(len(pool)), min(args.n, len(pool)))
    train_docs = [pool[i] for i in sampled_indices]


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    vocab_size = len(tokenizer.word_index) + 1

    train_seqs = build_sequences_from_docs(
        train_docs, tokenizer, args.seq_length, args.max_tokens_per_doc, args.max_total_sequences
    )
    val_seqs = build_sequences_from_docs(
        val_docs, tokenizer, args.seq_length, args.max_tokens_per_doc, args.max_total_sequences
    )

    if len(train_seqs) == 0 or len(val_seqs) == 0:
        print(
            f"n={args.n} seed={args.seed} -> Not enough sequences. "
            f"train_seqs={len(train_seqs)} val_seqs={len(val_seqs)}"
        )
        return

    X_train, y_train = train_seqs[:, :-1], train_seqs[:, -1]
    X_val, y_val = val_seqs[:, :-1], val_seqs[:, -1]

    model = Sequential()
    model.add(Embedding(vocab_size, 50))
    model.add(LSTM(100))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(25)],
    )

    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
    loss, acc = model.evaluate(X_val, y_val, verbose=0) #change that to sparse top cateogeorical accuracy

    print(f"n={args.n} seed={args.seed} val_loss={loss:.4f} val_acc={acc:.4f}")


if __name__ == "__main__":
    main()