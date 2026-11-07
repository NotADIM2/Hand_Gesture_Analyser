"""Turn signed words into natural sentences (sign words mode).

Sign languages have their own grammar: "ME HELP" means "I need help", "YOU GOOD" is
the question "Are you good?". Words signed in a row make a phrase; when the signer
pauses (PHRASE_PAUSE_SECONDS), the phrase is turned into an English sentence with the
table below, matching the longest known sequence first, and spoken. Words that don't
start a known sequence are said on their own ("HELLO" -> "Hello!").
"""
import config

PHRASES: dict[tuple[str, ...], str] = {
    # Needs and help
    ("ME", "HELP"): "I need help.",
    ("HELP", "ME"): "Please help me.",
    ("YOU", "HELP", "ME"): "Can you help me?",
    ("HELP", "YOU"): "Can I help you?",
    ("ME", "WATER"): "I would like some water.",
    ("ME", "WATER", "PLEASE"): "I would like some water, please.",
    ("WATER", "PLEASE"): "Water, please.",
    ("MORE", "WATER"): "More water, please.",
    ("MORE", "PLEASE"): "More, please.",
    ("ME", "WAIT"): "I will wait.",
    ("YOU", "WAIT"): "Please wait.",
    # Feelings
    ("ME", "GOOD"): "I am good.",
    ("ME", "BAD"): "I feel bad.",
    ("ME", "SORRY"): "I am sorry.",
    ("YOU", "GOOD"): "Are you good?",
    ("YOU", "BAD"): "Are you feeling bad?",
    ("ME", "SAME"): "Me too.",
    ("SAME", "ME"): "Me too.",
    ("YOU", "SAME"): "You too.",
    # Questions
    ("WHO", "YOU"): "Who are you?",
    ("WHERE", "YOU"): "Where are you?",
    ("WHERE", "YOU", "GO"): "Where are you going?",
    ("WHERE", "HOME"): "Where is home?",
    ("WHERE", "WATER"): "Where is the water?",
    ("WHAT", "YOU"): "What about you?",
    ("YOU", "HOME"): "Are you home?",
    # Going
    ("ME", "GO"): "I am going.",
    ("ME", "GO", "HOME"): "I am going home.",
    ("ME", "HOME"): "I am going home.",
    ("YOU", "GO"): "You go.",
    ("YOU", "GO", "HOME"): "Go home.",
    ("GO", "HOME"): "Go home.",
    # Greetings
    ("HELLO", "YOU"): "Hello to you!",
    ("THANK YOU", "MORE"): "Thank you very much.",
    ("YES", "PLEASE"): "Yes, please.",
    ("NO", "THANK YOU"): "No, thank you.",
    ("STOP", "DANGER"): "Stop, it's dangerous!",
    ("DANGER", "GO"): "Danger, get out!",
}

SINGLE_WORDS: dict[str, str] = {
    "HELLO": "Hello!", "GOODBYE": "Goodbye!", "PLEASE": "Please.", "THANK YOU": "Thank you.",
    "SORRY": "Sorry.", "YES": "Yes.", "NO": "No.", "HELP": "Help!", "WHO": "Who?", "WHAT": "What?",
    "WHERE": "Where?", "GOOD": "Good.", "BAD": "Bad.", "ME": "Me.", "YOU": "You.", "SAME": "Same.",
    "MORE": "More.", "HOME": "Home.", "WATER": "Water.", "STOP": "Stop!", "WAIT": "Wait.", "GO": "Go!",
    "DANGER": "Danger!",
}

LONGEST = max(len(k) for k in PHRASES)


def to_sentence(words: list[str]) -> str:
    """["ME", "HELP"] -> "I need help."  Longest known sequence first, left to right."""
    parts = []
    i = 0
    while i < len(words):
        for n in range(min(LONGEST, len(words) - i), 1, -1):
            phrase = PHRASES.get(tuple(words[i:i + n]))
            if phrase:
                parts.append(phrase)
                i += n
                break
        else:
            parts.append(single(words[i]))
            i += 1
    return " ".join(parts)


def single(word: str) -> str:
    return SINGLE_WORDS.get(word, word.capitalize() + ".")


class PhraseBuilder:
    """Collects signed words; after a pause, hands back the phrase as a sentence."""

    def __init__(self, pause: float = config.PHRASE_PAUSE_SECONDS):
        self.pause = pause
        self.words: list[str] = []
        self.sentence = ""          # the last finished sentence (shown until the next word)
        self.is_phrase = False      # it joined words into something new ("I need help."), so say it
        self._last_at = 0.0

    def add(self, word: str, now: float) -> None:
        self.words.append(word)
        self._last_at = now

    def update(self, now: float) -> str | None:
        """Call once per frame; returns the sentence when a phrase is finished."""
        if not self.words or now - self._last_at < self.pause:
            return None
        self.sentence = to_sentence(self.words)
        self.is_phrase = self.sentence != " ".join(single(w) for w in self.words)
        self.words = []
        return self.sentence

    @property
    def text(self) -> str:
        """What to show: the words being signed, or the last finished sentence."""
        return " ".join(self.words) if self.words else self.sentence

    def clear(self) -> None:
        self.words = []
        self.sentence = ""
        self.is_phrase = False
