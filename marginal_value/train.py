import os
import argparse
import json
import random
import re
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('ERROR')

def clean_text(doc):
    doc = doc.replace("--", " ")
    doc = doc.lower()
    doc = re.sub(r"[^a-z\s]+", " ", doc)
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc

def load_docs(path):
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

def build_sequences_from_docs(docs, tokenizer, seq_length, max_tokens_per_doc, max_total_sequences):
    sequences = []
    total = 0
    for d in docs:
        if not d:
            continue
        ids = tokenizer.texts_to_sequences([d])[0]
        if not ids:
            continue
        if max_tokens_per_doc and len(ids) > max_tokens_per_doc:
            ids = ids[:max_tokens_per_doc]
        if len(ids) <= seq_length:
            continue
        for i in range(seq_length, len(ids)):
            sequences.append(ids[i - seq_length: i + 1])
            total += 1
            if max_total_sequences and total >= max_total_sequences:
                return np.array(sequences, dtype=np.int32)
    return np.array(sequences, dtype=np.int32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="20000_openwebtext.json")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_tokens_per_doc", type=int, default=200)
    parser.add_argument("--max_total_sequences", type=int, default=50000)
    args = parser.parse_args()

    if args.n is None:
        args.n = int(input("Enter n: ").strip())

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    raw_docs = load_docs(args.data_path)
    cleaned_docs = [clean_text(t) for t in raw_docs if isinstance(t, str) and t.strip()]
    np.random.shuffle(cleaned_docs)
    m = len(cleaned_docs)

    holdout = max(1, int(0.1 * m))
    val_docs = cleaned_docs[:holdout]
    pool = cleaned_docs[holdout:]
    sampled_indices = random.sample(range(len(pool)), min(args.n, len(pool)))
    train_docs = [pool[i] for i in sampled_indices]

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(pool)
    vocab_size = 10000

    train_seqs = build_sequences_from_docs(train_docs, tokenizer, args.seq_length, args.max_tokens_per_doc, args.max_total_sequences)
    val_seqs = build_sequences_from_docs(val_docs, tokenizer, args.seq_length, args.max_tokens_per_doc, args.max_total_sequences)

    if len(train_seqs) == 0 or len(val_seqs) == 0:
        print(f"n={args.n} seed={args.seed} train_loss=NA train_acc=NA val_loss=NA val_acc=NA")
        return

    X_train, y_train = train_seqs[:, :-1], train_seqs[:, -1]
    X_val, y_val = val_seqs[:, :-1], val_seqs[:, -1]

    model = Sequential([
        Embedding(vocab_size, 64),
        LSTM(64),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(5)],
    )

    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
    train_loss = history.history['loss'][-1]
    train_acc = history.history['sparse_top_k_categorical_accuracy'][-1]
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"n={args.n} seed={args.seed} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

if __name__ == "__main__":
    main()