import csv
from pathlib import Path

from .rules import (
    additional_question_rules,
    additional_statement_rules,
    basic_statement,
    do_not_statement,
    i_am_statement,
    i_feeling_statement,
    i_state_statement,
    i_statement_rules,
    meeting_rules,
    simple_phrase_rules,
    wh_question,
    where_question,
    who_question,
    why_question,
    you_question,
    you_state_statement,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "gloss_dataset.csv"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def gloss_to_sentence(gloss):
    gloss = gloss.strip().upper()

    for row in load_dataset():
        if row["SIGN GLOSSES"].strip().upper() == gloss:
            return row["Sentence"].strip()

    return None


def translate_gloss(gloss):
    gloss = gloss.strip().upper()

    # Specific statement rules FIRST.
    # These must run before generic rules.
    result = additional_statement_rules(gloss)
    if result:
        return result

    # Specific question rules.
    result = where_question(gloss)
    if result:
        return result

    result = why_question(gloss)
    if result:
        return result

    result = who_question(gloss)
    if result:
        return result

    result = you_question(gloss)
    if result:
        return result

    result = additional_question_rules(gloss)
    if result:
        return result

    # Generic WH-question rule.
    result = wh_question(gloss)
    if result:
        return result

    result = meeting_rules(gloss)
    if result:
        return result

    # Specific I-statement rules.
    result = i_statement_rules(gloss)
    if result:
        return result

    result = additional_statement_rules(gloss)
    if result:
        return result

    result = do_not_statement(gloss)
    if result:
        return result

    result = i_state_statement(gloss)
    if result:
        return result

    result = i_feeling_statement(gloss)
    if result:
        return result

    result = i_am_statement(gloss)
    if result:
        return result

    result = you_state_statement(gloss)
    if result:
        return result

    result = simple_phrase_rules(gloss)
    if result:
        return result

    result = simple_phrase_rules(gloss)
    if result:
        return result

    result = meeting_rules(gloss)
    if result:
        return result

    return basic_statement(gloss)

    # Generic fallback.
    return basic_statement(gloss)


if __name__ == "__main__":
    data = load_dataset()

    print("Number of sentence-gloss pairs:", len(data))
    print("First pair:")
    print(data[0])

    test_glosses = [
        "YOU FREE TODAY",
        "YOU HIDE SOMETHING",
        "DO NOT TAKE IT HEART",
        "HE ON WAY",
        "HI HOW YOU",
    ]

    print("\nTest translations:")

    for gloss in test_glosses:
        result = translate_gloss(gloss)
        print(f"{gloss} -> {result}")