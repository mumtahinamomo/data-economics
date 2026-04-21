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
    default_data_path = os.path.join(os.path.dirname(__file__), "2500_openwebtext.json")
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

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # tf.random.set_seed(args.seed)

    raw_docs = load_docs(args.data_path)
    if not raw_docs:
        raise ValueError(f"No documents loaded from: {args.data_path}")

    cleaned_docs = [clean_text(t) for t in raw_docs if isinstance(t, str) and t.strip()]
    np.random.shuffle(cleaned_docs)
    m = len(cleaned_docs)
    if m < 2:
        raise ValueError("Need at least 2 documents total.")

    print("Total number of docs is " + str(m))

    holdout = max(1, int(0.1 * m))
    val_docs = cleaned_docs[:holdout]
    pool = cleaned_docs[holdout:]
    sampled_indices = random.sample(range(len(pool)), min(args.n, len(pool)))
    train_docs = [pool[i] for i in sampled_indices]
    print("train docs size is ", str(len(train_docs)))
    print("val docs size is ", str(len(val_docs)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(pool)
    print("Some token stats:")
    total_tokens = sum(tokenizer.word_counts.values())
    top_10_words = [word for word, index in tokenizer.word_index.items() if index <= 10]
    top_10_counts = [tokenizer.word_counts[word] for word in top_10_words]
    print(f"Total Tokens: {total_tokens}")
    for (i, w) in enumerate(top_10_words):
        proportion = (top_10_counts[i] / total_tokens) * 100
        print(f"{w}: Appears {top_10_counts[i]} times, {proportion:.2f}% of the dataset")

    print("")

    vocab_size = len(tokenizer.word_index) + 1

    train_seqs = build_sequences_from_docs(
        train_docs, tokenizer, args.seq_length, args.max_tokens_per_doc, args.max_total_sequences
    )
    val_seqs = build_sequences_from_docs(
        val_docs, tokenizer, args.seq_length, args.max_tokens_per_doc, args.max_total_sequences
    )
    # print("val_seqs[0] is ", val_seqs[0])
    if len(train_seqs) == 0 or len(val_seqs) == 0:
        print(
            f"n={args.n} seed={args.seed} -> Not enough sequences. "
            f"train_seqs={len(train_seqs)} val_seqs={len(val_seqs)}"
        )
        return

    X_train, y_train = train_seqs[:, :-1], train_seqs[:, -1]
    X_val, y_val = val_seqs[:, :-1], val_seqs[:, -1]

    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(100))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(25)],
    )

    print("before fitting, X_train shape is ", X_train.shape, " and y_train shape is ", y_train.shape)
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    print("before fitting, X_val shape is ", X_val.shape, " and y_val shape is ", y_val.shape)
    loss, acc = model.evaluate(X_val, y_val, verbose=1) #change that to sparse top cateogeorical accuracy

    print("DEBUGGING")
    sample_seq = X_val[0] 
    debug_predictions(model, tokenizer, sample_seq)
   
    print(f"n={args.n} seed={args.seed} val_loss={loss:.4f} val_acc={acc:.4f}")


def debug_predictions(model, tokenizer, sequence, window_size=5, top_n=5):
    # Reverse the word_index to lookup words by their integer ID
    id_to_word = {v: k for k, v in tokenizer.word_index.items()}
    
    print(f"{'Context Sequence':<40} | {'Top Predictions (Word: Prob)'}")
    print("-" * 100)

    # We start predicting once we have enough words for the window_size
    for i in range(len(sequence) - 1):
        # 1. Prepare the input window
        # Slice the sequence and pad if it's shorter than window_size
        current_context = sequence[max(0, i - window_size + 1) : i + 1]
        input_data = tf.keras.preprocessing.sequence.pad_sequences(
            [current_context], maxlen=window_size, padding='pre'
        )

        # 2. Get model prediction (Softmax distribution)
        preds = model.predict(input_data, verbose=0)[0]

        # 3. Extract Top N indices and values
        top_probs, top_indices = tf.math.top_k(preds, k=top_n)

        # 4. Format the output
        context_words = " ".join([id_to_word.get(idx, "?") for idx in current_context])
        prediction_list = []
        
        for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
            word = id_to_word.get(idx, "<UNK>")
            prediction_list.append(f"{word}: {prob:.2%}")

        print(f"{context_words[-40:]:>40} | {', '.join(prediction_list)}")

# Usage:
# sample_seq = validation_sequences[0] 
# debug_predictions(my_model, my_tokenizer, sample_seq)

if __name__ == "__main__":
    main()