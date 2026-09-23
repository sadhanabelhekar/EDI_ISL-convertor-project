from rule_based.gloss_to_sentence import translate_gloss


def gloss_to_english(gloss):
    """
    Convert ISL gloss into a natural English sentence.
    """

    if not gloss or not gloss.strip():
        return ""

    return translate_gloss(gloss)


if __name__ == "__main__":
    test_glosses = [
        "YOU FREE TODAY",
        "WHERE YOU FROM",
        "WHY YOU CRY",
        "I HELP YOU",
        "BRING WATER ME",
        "DO NOT TAKE IT HEART",
        "HE ON WAY",
        "NICE MEET YOU",
    ]

    for gloss in test_glosses:
        result = gloss_to_english(gloss)

        print(f"Gloss   : {gloss}")
        print(f"English : {result}")
        print("-" * 40)