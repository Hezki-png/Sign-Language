"""Builds the running sentence from typed letters and recognised signs."""
from .config import WORDS_FILE


def load_words():
    if not WORDS_FILE.exists():
        return []
    return [w.strip().upper() for w in WORDS_FILE.read_text(encoding="utf-8").split() if len(w.strip()) > 1]


class Composer:
    def __init__(self, vocabulary=()):
        self.vocabulary = list(vocabulary)  # most frequent first
        self.words = []
        self.partial = ""  # the word currently being fingerspelled

    def add_letter(self, letter):
        self.partial += letter.upper()

    def add_word(self, word):
        self.end_word()
        self.words.append(word.upper())

    def end_word(self):
        if self.partial:
            self.words.append(self.partial)
            self.partial = ""

    def backspace(self):
        if self.partial:
            self.partial = self.partial[:-1]
        elif self.words:
            self.words.pop()

    def clear(self):
        self.words, self.partial = [], ""

    def suggestions(self, k=3):
        if not self.partial:
            return []
        out = []
        for word in self.vocabulary:
            if word.startswith(self.partial) and word != self.partial:
                out.append(word)
                if len(out) == k:
                    break
        return out

    def accept(self, index):
        options = self.suggestions()
        if index < len(options):
            self.partial = ""
            self.words.append(options[index])

    @property
    def text(self):
        return " ".join(self.words + ([self.partial] if self.partial else []))
