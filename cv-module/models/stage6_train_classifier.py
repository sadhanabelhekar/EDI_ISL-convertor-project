"""
stage6_train_classifier.py

Trains a first baseline classifier (Random Forest) on the landmark data
extracted in Stage 4, and reports honest accuracy on unseen test data.

WHERE TO PUT THIS FILE:
    cv-module/models/stage6_train_classifier.py

HOW TO RUN (from inside cv-module/, with venv activated):
    python models/stage6_train_classifier.py

WHAT IT DOES:
    1. Loads each word's .npy file from models/landmarks_data/
    2. Combines them into one big table: X = landmark numbers, y = word labels
    3. Splits into train/test sets (80% train, 20% test — test set is NEVER
       shown to the model during training, so accuracy on it is honest)
    4. Trains a Random Forest classifier
    5. Prints overall accuracy + a per-word breakdown (confusion matrix info)
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LANDMARKS_DIR = "models/landmarks_data"

# Minimum successful samples a word needs to be included in training.
# Words below this threshold don't have enough examples for the model
# to learn a reliable pattern — they're logged separately, not silently dropped.
MIN_SAMPLES_PER_WORD = 8


def load_dataset():
    X = []
    y = []
    included_words = []
    excluded_words = []

    for filename in sorted(os.listdir(LANDMARKS_DIR)):
        if not filename.endswith(".npy"):
            continue
        word = filename[:-4]  # strip ".npy"
        path = os.path.join(LANDMARKS_DIR, filename)
        data = np.load(path)  # shape: (num_samples, 63)

        if len(data) < MIN_SAMPLES_PER_WORD:
            excluded_words.append((word, len(data)))
            continue

        included_words.append((word, len(data)))
        for row in data:
            X.append(row)
            y.append(word)

    print(f"Included {len(included_words)} words (>= {MIN_SAMPLES_PER_WORD} samples each)")
    print(f"Excluded {len(excluded_words)} words (too few samples):")
    for word, count in excluded_words:
        print(f"    {word}: only {count} samples")
    print()

    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()
    print(f"Total samples loaded: {len(X)}")
    print(f"Words: {sorted(set(y))}\n")

    # 80% train, 20% test. random_state=42 just makes the split reproducible
    # so you get the same split every time you rerun this script.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} unseen samples\n")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"=== ACCURACY ON UNSEEN TEST DATA: {accuracy * 100:.1f}% ===\n")

    print("Per-word breakdown (precision/recall):")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows = actual word, columns = predicted word):")
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, predictions, labels=labels)
    print("Labels order:", labels)
    print(cm)


if __name__ == "__main__":
    main()