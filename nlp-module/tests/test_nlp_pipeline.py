from pathlib import Path
import sys

# Add nlp-module to Python path
NLP_MODULE_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NLP_MODULE_PATH))

from nlp_pipeline import gloss_to_english


def test_you_free_today():
    assert gloss_to_english("YOU FREE TODAY") == "Are you free today?"


def test_where_question():
    assert gloss_to_english("WHERE YOU FROM") == "Where are you from?"


def test_why_question():
    assert gloss_to_english("WHY YOU CRY") == "Why are you crying?"


def test_i_help_you():
    assert gloss_to_english("I HELP YOU") == "Can I help you?"


def test_bring_water():
    assert gloss_to_english("BRING WATER ME") == "Bring water for me."


def test_do_not_take_it_heart():
    assert gloss_to_english("DO NOT TAKE IT HEART") == "Do not take it to the heart."


def test_he_on_way():
    assert gloss_to_english("HE ON WAY") == "He is on the way."


def test_nice_meet_you():
    assert gloss_to_english("NICE MEET YOU") == "Nice to meet you."


def test_empty_input():
    assert gloss_to_english("") == ""


def test_whitespace_input():
    assert gloss_to_english("   ") == ""