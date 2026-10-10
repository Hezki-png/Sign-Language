"""Runtime decision logic: turn per-frame classifier output into typed letters and signs."""
import joblib
import numpy as np

from . import config as cfg
from .features import clip_features, motion, static_features


def load_model(path):
    """A trained bundle ({"pipeline", "labels"}) or None if it has not been trained yet."""
    return joblib.load(path) if path.exists() else None


class LetterRecognizer:
    """Types a letter once it has been held still and confidently for LETTER_HOLD seconds."""

    def __init__(self, bundle):
        self.pipeline = bundle["pipeline"]
        self.labels = [str(c) for c in self.pipeline.classes_]
        self.reset()

    def reset(self):
        self._probs = None
        self._candidate = None
        self._since = 0.0
        self._typed = False

    def update(self, lm, t, still):
        """Returns (label, confidence, hold progress 0..1, letter typed this frame or None)."""
        probs = self.pipeline.predict_proba(static_features(lm)[None])[0]
        self._probs = probs if self._probs is None else 0.5 * self._probs + 0.5 * probs
        top = int(np.argmax(self._probs))
        label, conf = self.labels[top], float(self._probs[top])

        if conf < cfg.LETTER_MIN_CONF or not still:
            self._candidate = None
            return label, conf, 0.0, None
        if label != self._candidate:
            self._candidate, self._since, self._typed = label, t, False
            return label, conf, 0.0, None

        need = cfg.LETTER_REPEAT if self._typed else cfg.LETTER_HOLD
        held = t - self._since
        if held >= need:
            self._since, self._typed = t, True
            return label, conf, 1.0, label
        return label, conf, held / need, None


class SignRecognizer:
    """Classifies the sliding window and fires when a moving sign is seen consistently."""

    def __init__(self, bundle):
        self.pipeline = bundle["pipeline"]
        self.labels = [str(c) for c in self.pipeline.classes_]
        self._candidate = None
        self._streak = 0

    def update(self, buffer):
        """Returns (label, confidence, sign fired this frame or None). Clears the buffer on fire."""
        if buffer.span < 0.9 * cfg.CLIP_SECONDS:
            return None, 0.0, None
        t, hands, present = buffer.arrays()
        if motion(t, hands, present, cfg.CLIP_SECONDS) < cfg.SIGN_MIN_MOTION:
            self._streak = 0
            return None, 0.0, None

        probs = self.pipeline.predict_proba(clip_features(t, hands, present)[None])[0]
        top = int(np.argmax(probs))
        label, conf = self.labels[top], float(probs[top])
        if label == cfg.IDLE or conf < cfg.SIGN_MIN_CONF:
            self._streak = 0
            return None, conf, None

        self._streak = self._streak + 1 if label == self._candidate else 1
        self._candidate = label
        if self._streak >= cfg.SIGN_STREAK:
            self._streak = 0
            buffer.clear()  # the same motion must not fire twice
            return label, conf, label
        return label, conf, None
