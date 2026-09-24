"""
stage8_train_lstm.py

Stage 8b — trains an LSTM classifier on the landmark SEQUENCES extracted
in Stage 8a. This is your first real continuous/motion-aware recognition
result, distinct from Stage 6's single-frame word classifier.

WHERE TO PUT THIS FILE:
    cv-module/models/stage8_train_lstm.py

BEFORE RUNNING (one-time):
    pip install tensorflow scikit-learn

HOW TO RUN (from inside cv-module/, with venv activated):
    python models/stage8_train_lstm.py

WHAT IT PRODUCES:
    Prints accuracy on unseen test sequences + a per-sentence breakdown,
    same style as Stage 6's classification_report.
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

SEQUENCE_DIR = "models/sequence_data"
MAX_SEQ_LEN = 20   # must match stage8_extract_sequences.py
NUM_FEATURES = 63  # 21 landmarks x 3 coords

# Same reasoning as Stage 6: sentences with very few takes can't be learned
# reliably. Require at least this many takes before including a sentence.
MIN_TAKES_PER_SENTENCE = 4


def load_dataset():
    with open(os.path.join(SEQUENCE_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels_map = json.load(f)

    # group filenames by sentence text so we can filter by take count
    by_sentence = {}
    for filename, sentence in labels_map.items():
        by_sentence.setdefault(sentence, []).append(filename)

    X, y = [], []
    excluded = []

    for sentence, filenames in by_sentence.items():
        if len(filenames) < MIN_TAKES_PER_SENTENCE:
            excluded.append((sentence, len(filenames)))
            continue
        for filename in filenames:
            seq = np.load(os.path.join(SEQUENCE_DIR, filename))
            X.append(seq)
            y.append(sentence)

    print(f"Included {len(by_sentence) - len(excluded)} sentences "
          f"(>= {MIN_TAKES_PER_SENTENCE} takes each)")
    print(f"Excluded {len(excluded)} sentences (too few takes)")
    if excluded:
        for sentence, count in excluded[:10]:
            print(f"    '{sentence}': only {count} takes")
        if len(excluded) > 10:
            print(f"    ... and {len(excluded) - 10} more")
    print()

    return np.array(X), np.array(y)


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(MAX_SEQ_LEN, NUM_FEATURES)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    X, y_text = load_dataset()
    print(f"Total sequences loaded: {len(X)}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    num_classes = len(encoder.classes_)
    print(f"Number of sentence classes: {num_classes}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} sequences, testing on {len(X_test)} unseen sequences\n")

    model = build_model(num_classes)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== ACCURACY ON UNSEEN TEST SEQUENCES: {accuracy * 100:.1f}% ===\n")

    print("Per-sentence breakdown:")
    print(classification_report(
        y_test, y_pred,
        labels=range(num_classes),
        target_names=encoder.classes_,
        zero_division=0,
    ))

    model.save("models/lstm_sentence_model.keras")
    print("Model saved to models/lstm_sentence_model.keras")


if __name__ == "__main__":
    main()