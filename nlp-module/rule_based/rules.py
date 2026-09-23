def do_not_statement(gloss):
    words = gloss.strip().upper().split()

    if not words:
        return None

    if words[0] == "DONOT":
        words[0] = "DO"
        words.insert(1, "NOT")

    if words[0] == "DONT":
        words[0] = "DO"
        words.insert(1, "NOT")

    if words[0] != "DO" or len(words) < 3 or words[1] != "NOT":
        return None

    sentence = " ".join(word.lower() for word in words)
    return sentence.capitalize() + "."


def i_state_statement(gloss):
    words = gloss.strip().upper().split()

    if len(words) < 2 or words[0] != "I":
        return None

    states = {
        "HUNGRY",
        "TIRED",
        "HAPPY",
        "BORED",
        "COLD",
        "AFRAID",
    }

    if words[1] in states:
        rest = " ".join(word.lower() for word in words[1:])
        return f"I am {rest}."

    if len(words) >= 3 and words[1] in {"VERY", "REALLY"} and words[2] in states:
        rest = " ".join(word.lower() for word in words[1:])
        return f"I am {rest}."

    return None


def basic_statement(gloss):
    words = gloss.strip().upper().split()

    if len(words) < 2:
        return None

    sentence = " ".join(word.lower() for word in words)
    return sentence.capitalize() + "."


def where_question(gloss):
    words = gloss.strip().upper().split()

    if words == ["WHERE", "YOU", "FROM"]:
        return "Where are you from?"

    return None


def why_question(gloss):
    words = gloss.strip().upper().split()

    if words == ["WHY", "YOU", "CRY"]:
        return "Why are you crying?"

    if len(words) == 3 and words[0] == "WHY" and words[1] == "YOU":
        adjective = words[2].lower()
        return f"Why are you {adjective}?"

    return None

def who_question(gloss):
    words = gloss.strip().upper().split()

    if words == ["WHO", "YOU"]:
        return "Who are you?"

    return None


def wh_question(gloss):
    words = gloss.strip().upper().split()

    if words == ["HOW", "YOU"]:
        return "How are you?"
    
    if not words or words[0] not in {
        "WHAT",
        "WHERE",
        "WHO",
        "WHY",
        "WHEN",
        "HOW",
    }:
        return None

    sentence = " ".join(word.lower() for word in words)
    return sentence.capitalize() + "?"


def i_feeling_statement(gloss):
    words = gloss.strip().upper().split()

    if len(words) < 3 or words[0] != "I" or words[1] != "FEELING":
        return None

    rest = " ".join(word.lower() for word in words[2:])
    return f"I am feeling {rest}."


def you_state_statement(gloss):
    words = gloss.strip().upper().split()

    if len(words) != 2 or words[0] != "YOU":
        return None

    states = {
        "BAD",
        "GOOD",
        "WELCOME",
    }

    if words[1] in states:
        return f"You are {words[1].lower()}."

    return None


def i_am_statement(gloss):
    words = gloss.strip().upper().split()

    if len(words) < 2 or words[0] != "I":
        return None

    skip_words = {
        "HELP",
        "NEED",
        "PROMISE",
        "ENJOYED",
        "GOT",
        "STOPPED",
        "LIKE",
        "LOVE",
        "DO",
        "DONOT",
        "DONT",
        "CRY",
    }

    if words[1] in skip_words:
        return None

    rest = " ".join(word.lower() for word in words[1:])
    return f"I am {rest}."


def you_question(gloss):
    words = gloss.strip().upper().split()

    if not words or words[0] != "YOU":
        return None

    if words == ["YOU", "FREE", "TODAY"]:
        return "Are you free today?"

    if words == ["YOU", "HIDE", "SOMETHING"]:
        return "Are you hiding something?"

    return None


def simple_phrase_rules(gloss):
    words = gloss.strip().upper().split()

    if words == ["BRING", "WATER", "ME"]:
        return "Bring water for me."

    if words == ["COMB", "YOU", "HAIR"]:
        return "Comb your hair."

    if words == ["DO", "ME", "FAVOUR"]:
        return "Do me a favour."

    if words == ["GO", "SLEEP"]:
        return "Go and sleep."

    if words == ["PREPARE", "BED"]:
        return "Prepare the bed."

    if words == ["SERVE", "FOOD"]:
        return "Serve the food."

    if words == ["WEAR", "SHIRT"]:
        return "Wear the shirt."

    if len(words) == 3 and words[0] == "MY" and words[1] == "NAME":
        name = words[2]
        return f"My name is {name}."

    if words == ["NICE", "MEET", "YOU"]:
        return "Nice to meet you."

    if words == ["THIS", "PLACE", "BEAUTIFUL"]:
        return "This place is beautiful."

    if words == ["WE", "ALL", "WITH", "YOU"]:
        return "We are all with you."

    return None


def additional_question_rules(gloss):
    words = gloss.strip().upper().split()

    if words == ["YOU", "REPEAT", "PLEASE"]:
        return "Can you repeat that please?"

    if words == ["YOU", "PLEASE", "TALK", "SLOWER"]:
        return "Could you please talk slower?"

    if words == ["HOW", "THINGS"]:
        return "How are things?"

    if words == ["HOW", "I", "HELP", "YOU"]:
        return "Can I help you?"

    if words == ["HOW", "I", "TRUST", "YOU"]:
        return "How can I trust you?"

    if words == ["HOW", "OLD", "YOU"]:
        return "How old are you?"

    if words == ["THAT", "KIND", "YOU"]:
        return "That is so kind of you."

    if words == ["WHAT", "YOU", "DO"]:
        return "What are you doing?"

    if words == ["WHAT", "YOU", "TELL", "HIM"]:
        return "What did you tell him?"

    if words == ["WHAT", "DO", "YOU", "WANT", "BECOME"]:
        return "What do you want to become?"

    if words == ["WHAT", "HAVE", "YOU", "PLAN", "YOUR", "CAREER"]:
        return "What have you planned for your career?"

    if words == ["WHAT", "YOUR", "PHONE", "NUMBER"]:
        return "What is your phone number?"

    if words == ["WHEN", "TRAIN", "LEAVE"]:
        return "When will the train leave?"

    if words == ["WHICH", "COLLEGE", "SCHOOL", "YOU", "FROM"]:
        return "Which college school are you from?"

    if words == ["WHY", "YOU", "CRY"]:
        return "Why are you crying?"

    if words == ["YOU", "DO", "IT"]:
        return "You can do it."

    return None


def i_statement_rules(gloss):
    words = gloss.strip().upper().split()

    if words == ["I", "HELP", "YOU"]:
        return "Can I help you?"

    if words == ["I", "AFRAID", "THAT"]:
        return "I am afraid of that."

    if words == ["I", "CRY"]:
        return "I am crying."

    if words == ["I", "AM", "SO", "SORRY", "TO", "HEAR", "THAT"]:
        return "I am so sorry to hear that."

    if words == ["I", "NOT", "HELP", "YOU", "THERE"]:
        return "I can not help you there."

    if words == ["I", "DONOT", "AGREE"]:
        return "I do not agree."

    if words == ["I", "REALLY", "APPRECIATE", "IT"]:
        return "I really appreciate it."

    if words == ["I", "SOMEHOW", "GOT", "KNOW", "ABOUT", "IT"]:
        return "I somehow got to know about it."

    if words == ["I", "STOPPED", "BY", "SOMEONE"]:
        return "I was stopped by some one."

    return None


def additional_statement_rules(gloss):
    words = gloss.strip().upper().split()

    if words == ["CONGRATULATIIONS"]:
        return "Congratulations."

    if words == ["DO", "NOT", "TAKE", "IT", "HEART"]:
        return "Do not take it to the heart."

    if words == ["YOUR", "FOOD"]:
        return "Had your food."

    if words == ["HE", "CAME", "TRAIN"]:
        return "He came by train."

    if words == ["HE", "GO", "INTO", "ROOM"]:
        return "He is going into the room."

    if words == ["HE", "ON", "WAY"]:
        return "He is on the way."

    if words == ["HE", "SHE", "MY", "FRIEND"]:
        return "He she is my friend."

    if words == ["HE", "COMING", "TODAY"]:
        return "He would be coming today."

    if words == ["HI", "HOW", "YOU"]:
        return "Hi how are you"

    if words == ["IT", "DO", "NOT", "MAKE", "ANY", "DIFFERENCE", "TO", "ME"]:
        return "It does not make any difference to me."

    if words == ["IT", "NICE", "CHAT", "WITH", "YOU"]:
        return "It was nice chatting with you."

    if words == ["NO", "NEED", "WORRY", "DONT", "WORRY"]:
        return "No need to worry dont worry."

    if words == ["NOW", "ONWARDS", "HE", "NEVER", "HURT", "YOU"]:
        return "Now onwards he will never hurt you."

    if words == ["WE", "GO", "OUTSIDE"]:
        return "Shall we go outside?"

def meeting_rules(gloss):
    words = gloss.strip().upper().split()

    if words == ["SORRY"]:
        return "Sorry."

    if words == ["I", "UNDERSTAND"]:
        return "I understand."

    if words == ["I", "AGREE"]:
        return "I agree."
  
    return None