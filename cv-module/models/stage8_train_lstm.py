"""
stage8_train_lstm.py  (two-hands version)

Tests the LSTM on the 16 meeting sentences using 5 different
train/test splits and prints the average accuracy.

HOW TO RUN (from inside cv-module/, venv activated):
    python models/stage8_train_lstm.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

SEQUENCE_DIR = "models/sequence_data"
MAX_SEQ_LEN = 20
NUM_FEATURES = 126
AUG_COPIES = 8
EPOCHS = 60

ONLY_SENTENCES = [
    "thank you so much", "nice to meet you", "you are welcome",
    "how are things", "can i help you", "can you repeat that please",
    "could you please talk slower", "i do not agree", "what do you think",
    "what you want", "what happened", "what are you doing",
    "i really appreciate it", "tell me truth", "try to understand",
    "that is so kind of you",
]


def load_dataset():
    with open(os.path.join(SEQUENCE_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels_map = json.load(f)

    X, y = [], []
    for filename, sentence in labels_map.items():
        if sentence not in ONLY_SENTENCES:
            continue
        try:
            seq = np.load(os.path.join(SEQUENCE_DIR, filename))
        except ValueError:
            continue
        X.append(seq)
        y.append(sentence)

    return np.array(X), np.array(y)


def augment_sequence(seq, rng):
    real_frames = np.any(seq != 0, axis=1)
    pts = seq.reshape(MAX_SEQ_LEN, -1, 3).copy()

    angle = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts[:, :, :2] = pts[:, :, :2] @ rot.T

    pts *= rng.uniform(0.9, 1.1)
    pts += rng.normal(0, 0.005, pts.shape)

    out = pts.reshape(MAX_SEQ_LEN, NUM_FEATURES)
    out[~real_frames] = 0.0

    k = int(rng.integers(0, 3))
    if k > 0:
        out = np.vstack([out[k:], np.zeros((k, NUM_FEATURES))])
    return out


def augment_training_set(X_train, y_train, copies, seed):
    rng = np.random.default_rng(seed)
    X_new, y_new = [X_train], [y_train]
    for _ in range(copies):
        X_new.append(np.array([augment_sequence(s, rng) for s in X_train]))
        y_new.append(y_train)
    return np.concatenate(X_new), np.concatenate(y_new)


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(MAX_SEQ_LEN, NUM_FEATURES)),
        layers.Masking(mask_value=0.0),
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
    print(f"Total sequences loaded: {len(X)}  (shape per sequence: {X.shape[1:]})")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    num_classes = len(encoder.classes_)
    print(f"Number of sentence classes: {num_classes}")
    print(f"Random guessing would get: {100 / num_classes:.1f}%")
    print("One hand, even frames: average 17.3% (lowest 13.6%, highest 22.7%)\n")

    accs = []
    for seed in [1, 2, 3, 4, 5]:
        tf.keras.backend.clear_session()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        X_aug, y_aug = augment_training_set(X_train, y_train, AUG_COPIES, seed)

        model = build_model(num_classes)
        model.fit(X_aug, y_aug, epochs=EPOCHS, batch_size=32, verbose=0)

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        print(f"Split {seed}: {acc * 100:.1f}%  "
              f"({int(round(acc * len(y_test)))}/{len(y_test)} correct)")

    print(f"\nAverage: {np.mean(accs) * 100:.1f}%   "
          f"(lowest {np.min(accs) * 100:.1f}%, highest {np.max(accs) * 100:.1f}%)")


if __name__ == "__main__":
    main()