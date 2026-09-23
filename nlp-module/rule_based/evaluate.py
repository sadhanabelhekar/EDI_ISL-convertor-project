import csv
from pathlib import Path

from .gloss_to_sentence import translate_gloss


BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DATASET = BASE_DIR / "data" / "gloss_dataset.csv"
MEETING_DATASET = BASE_DIR / "data" / "meeting_gloss_dataset.csv"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize(text):
    if not text:
        return ""

    return text.strip().rstrip(".?!").lower()


def evaluate_dataset(dataset, dataset_name):
    total = len(dataset)
    exact_matches = 0
    failed = []

    for row in dataset:
        gloss = row["SIGN GLOSSES"].strip()
        expected = row["Sentence"].strip()
        predicted = translate_gloss(gloss)

        if normalize(predicted) == normalize(expected):
            exact_matches += 1
        else:
            failed.append((gloss, expected, predicted))

    accuracy = (exact_matches / total * 100) if total else 0

    print("=" * 60)
    print(dataset_name)
    print("=" * 60)
    print(f"Total examples : {total}")
    print(f"Exact matches  : {exact_matches}")
    print(f"Accuracy       : {accuracy:.2f}%")

    if failed:
        print("\nFailed examples:")

        for gloss, expected, predicted in failed:
            print(f"Gloss    : {gloss}")
            print(f"Expected : {expected}")
            print(f"Predicted: {predicted}")
            print("-" * 40)

    print()


def main():
    original_dataset = load_dataset(ORIGINAL_DATASET)
    meeting_dataset = load_dataset(MEETING_DATASET)

    evaluate_dataset(
        original_dataset,
        "ORIGINAL DATASET"
    )

    evaluate_dataset(
        meeting_dataset,
        "MEETING DATASET"
    )

    combined_dataset = original_dataset + meeting_dataset

    evaluate_dataset(
        combined_dataset,
        "COMBINED DATASET"
    )


if __name__ == "__main__":
    main()