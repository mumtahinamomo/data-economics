import os
import argparse
import json
import random
import re
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer


# ---------- Cleaning (MachineLearningMastery-ish) ----------
def clean_text(doc: str) -> str:
    doc = doc.replace("--", " ")
    doc = doc.lower()
    # keep letters + whitespace only
    doc = re.sub(r"[^a-z\s]+", " ", doc)
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc


# ---------- Load docs from JSON ----------
def load_docs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    # Accept either: list of {"text": "..."} or list of strings
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
    return texts


# ---------- Build (50 words -> next word) sequences ----------
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

        # tokenize doc into word ids
        ids = tokenizer.texts_to_sequences([d])[0]
        if not ids:
            continue

        # cap doc length to keep runtime manageable
        if max_tokens_per_doc is not None and len(ids) > max_tokens_per_doc:
            ids = ids[:max_tokens_per_doc]

        if len(ids) <= seq_length:
            continue

        # sliding window
        for i in range(seq_length, len(ids)):
            seq = ids[i - seq_length : i + 1]  # length seq_length+1
            sequences.append(seq)
            total += 1
            if max_total_sequences is not None and total >= max_total_sequences:
                return np.array(sequences, dtype=np.int32)

    return np.array(sequences, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    default_data_path = os.path.join(os.path.dirname(__file__), "openwebtext_500.json")
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--n", type=int, default=None, help="Number of documents to sample (with replacement).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)

    # runtime-safety caps (important for OpenWebText!)
    parser.add_argument("--max_tokens_per_doc", type=int, default=400, help="Cap tokens per doc for speed.")
    parser.add_argument("--max_total_sequences", type=int, default=200000, help="Cap total sequences for speed.")
    args = parser.parse_args()

    # If n not provided, ask interactively
    if args.n is None:
        args.n = int(input("Enter n (number of documents to sample): ").strip())

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # load and clean docs
    raw_docs = load_docs(args.data_path)
    if not raw_docs:
        raise ValueError(f"No documents loaded from: {args.data_path}")

    cleaned_docs = [clean_text(t) for t in raw_docs if isinstance(t, str) and t.strip()]
    m = len(cleaned_docs)
    if m < 2:
        raise ValueError("Need at least 2 documents total.")

    # sample WITH replacement
    sampled_indices = [random.randrange(m) for _ in range(args.n)]
    sampled_set = set(sampled_indices)

    train_docs = [cleaned_docs[i] for i in sampled_indices]  # with replacement
    val_docs = [cleaned_docs[i] for i in range(m) if i not in sampled_set]  # rest as validation

    if len(val_docs) == 0:
        # edge case if n is huge and sampled_set covers everything
        # fallback: hold out 10% of docs for validation
        holdout = max(1, int(0.1 * m))
        val_docs = cleaned_docs[:holdout]
        train_docs = cleaned_docs[holdout:]

    # fit tokenizer on TRAIN docs only (important)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    vocab_size = len(tokenizer.word_index) + 1

    # build sequences
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

    # model (slightly simpler to keep it faster; add 2nd LSTM if you want)
    model = Sequential()
    model.add(Embedding(vocab_size, 50))
    model.add(LSTM(100))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
    loss, acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"n={args.n} seed={args.seed} val_loss={loss:.4f} val_acc={acc:.4f}")


if __name__ == "__main__":
    main()