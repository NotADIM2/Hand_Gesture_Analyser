"""Signed words into sentences, and the sign practice mode."""
import random

import pytest

import config
from phrases import PHRASES, SINGLE_WORDS, PhraseBuilder, to_sentence
from sign_practice import SignPractice
from sign_words import HINTS, WORDS


# --- Phrases --------------------------------------------------------------------------

@pytest.mark.parametrize("words, sentence", [
    (["ME", "HELP"], "I need help."),
    (["YOU", "GOOD"], "Are you good?"),
    (["WHERE", "YOU", "GO"], "Where are you going?"),        # longest match wins over WHERE YOU
    (["WHERE", "YOU"], "Where are you?"),
    (["ME", "WATER", "PLEASE"], "I would like some water, please."),
    (["HELLO", "ME", "HELP"], "Hello! I need help."),        # a greeting, then a phrase
    (["HELLO"], "Hello!"),
    (["THANK YOU"], "Thank you."),
    (["GO", "YES"], "Go! Yes."),                             # no phrase: word by word
    ([], ""),
])
def test_signed_words_become_sentences(words, sentence):
    assert to_sentence(words) == sentence


def test_every_word_and_phrase_uses_known_words():
    assert set(SINGLE_WORDS) == set(WORDS)
    assert all(set(k) <= set(WORDS) and len(k) >= 2 for k in PHRASES)


def test_a_pause_finishes_the_phrase():
    builder = PhraseBuilder(pause=2.0)
    builder.add("ME", 0.0)
    builder.add("HELP", 1.5)
    assert builder.text == "ME HELP"
    assert builder.update(3.0) is None                 # still signing
    assert builder.update(3.5) == "I need help."
    assert builder.is_phrase                           # joined into something new: say it
    assert builder.text == "I need help."              # shown until the next word
    assert builder.update(10.0) is None                # finished once


def test_single_words_are_not_said_twice():
    builder = PhraseBuilder(pause=2.0)
    builder.add("HELLO", 0.0)
    assert builder.update(2.0) == "Hello!" and not builder.is_phrase   # already spoken as a word
    builder.add("YOU", 3.0)
    assert builder.text == "YOU"
    builder.clear()
    assert builder.text == "" and builder.update(99.0) is None


# --- Practice -------------------------------------------------------------------------

def practice(words=WORDS):
    return SignPractice(words, rng=random.Random(1))


def test_correct_sign_scores_and_moves_on():
    p = practice()
    first = p.start(0.0)
    assert first == f"SIGN: {p.target}"
    target = p.target
    message = p.check(target, 2.5)
    assert message == "CORRECT!" and p.target != target
    assert (p.correct, p.tries, p.streak, p.best_streak) == (1, 1, 1, 1)
    assert p.attempts[0].seconds == 2.5


def test_wrong_sign_says_what_it_saw_and_resets_the_streak():
    p = practice(("HELLO", "YES"))
    p.start(0.0)
    p.check(p.target, 1.0)                       # one right
    target = p.target
    wrong = "YES" if target == "HELLO" else "HELLO"
    assert p.check(wrong, 2.0) == f"THAT WAS {wrong}, TRY AGAIN"
    assert p.target == target and p.streak == 0 and p.best_streak == 1
    assert p.summary() == "1/2 signs correct, best streak 1"


def test_skip_and_nothing_signed():
    p = practice()
    p.start(0.0)
    assert p.check(None, 1.0) is None and p.tries == 0
    target = p.target
    assert p.skip(5.0).startswith("SIGN: ") and p.target != target
    assert p.tries == 0 and len(p.attempts) == 1 and not p.attempts[0].correct


def test_every_word_once_before_repeats_and_never_twice_in_a_row():
    p = practice()
    p.start(0.0)
    asked = [p.target]
    for i in range(3 * len(WORDS)):
        p.skip(float(i))
        asked.append(p.target)
    assert sorted(asked[:len(WORDS)]) == sorted(WORDS)
    assert all(a != b for a, b in zip(asked, asked[1:]))


def test_status_shows_hint_score_and_streak():
    p = practice()
    p.start(0.0)
    lines = p.status_lines()
    assert lines[0] == f"HOW: {HINTS[p.target]}"
    assert "SCORE: 0 correct of 0 tried" in lines[1] and "n: skip" in lines[2]
    assert set(HINTS) == set(WORDS)


def test_practice_round_saved_with_exercise_results(tmp_path):
    import app
    from database import GestureDatabase

    p = practice()
    p.start(100.0)
    p.check(p.target, 103.0)
    p.check("NOT IT", 105.0)
    with GestureDatabase(tmp_path / "h.db") as db:
        app.save_practice(db, p, 130.0, "s1")
        app.save_practice(db, practice(), 130.0, "s1")   # nothing tried: not saved
        db.flush()
        saved = db.get_exercises()
    assert [(e.exercise, e.reps, e.seconds, e.detail) for e in saved] == [
        ("SIGN PRACTICE", 1, 30.0, "1/2 signs correct, best streak 1")]


def test_phrase_pause_setting():
    assert config.PHRASE_PAUSE_SECONDS > config.SIGN_COOLDOWN_SECONDS
