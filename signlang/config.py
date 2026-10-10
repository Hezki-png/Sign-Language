"""Paths and tunable thresholds shared by the app, the collector and the trainer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

HAND_MODEL = MODELS / "hand_landmarker.task"
LETTER_MODEL = MODELS / "letters.joblib"
SIGN_MODEL = MODELS / "signs.joblib"
STATIC_DATA = DATA / "static"    # static/<source>/<LETTER>.npy, each (N, 21, 3)
DYNAMIC_DATA = DATA / "dynamic"  # dynamic/<SIGN>/<timestamp>.npz
WORDS_FILE = DATA / "words.txt"  # one word per line, most frequent first

# ASL fingerspelling without the two letters that need motion. J and Z are
# handled as dynamic signs instead.
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
DEFAULT_SIGNS = ["HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "HELP", "SORRY",
                 "GOOD", "BAD", "LOVE", "EAT", "DRINK", "MORE", "STOP", "J", "Z"]
IDLE = "_idle_"  # "nothing is being signed" class for the dynamic model

# --- dynamic signs ---
CLIP_SECONDS = 1.5       # length of a recorded clip and of the live sliding window
CLIP_FRAMES = 20         # every clip is resampled to this many frames
SIGN_MIN_CONF = 0.80
SIGN_MIN_MOTION = 1.5    # landmark travel over the window, in palm lengths
SIGN_STREAK = 4          # consecutive agreeing frames before a sign is accepted

# --- fingerspelling ---
LETTER_MIN_CONF = 0.70
LETTER_HOLD = 0.6        # seconds a letter must be held to be typed
LETTER_REPEAT = 1.4      # keep holding this long to type the same letter again
STILL_WINDOW = 0.4       # seconds
STILL_MAX_MOTION = 0.6   # landmark travel over STILL_WINDOW, in palm lengths
LETTER_BLOCK_AFTER_SIGN = 1.5  # seconds letters are ignored after a sign fires

WORD_GAP = 1.5           # seconds without a hand that ends the current word
