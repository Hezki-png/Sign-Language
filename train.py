"""Train the letter and sign classifiers from everything under data/.

    python train.py            train whichever of the two has data
    python train.py letters    only the fingerspelling model
    python train.py signs      only the dynamic-sign model
"""
import argparse

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signlang import config as cfg
from signlang.features import clip_features, rotate_xy, static_features

USER_WEIGHT = 3      # your own letter samples count this many times
CLIP_VARIANTS = 20   # augmented copies per recorded clip
MIN_CLIPS = 5


def fit(x, y, hidden):
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden, max_iter=400, early_stopping=len(y) > 2000,
                                                          random_state=0))
    return model.fit(x, y)


def report(name, model, x, y):
    pred = model.predict(x)
    print(f"{name}: held-out accuracy {np.mean(pred == y):.1%} on {len(y)} samples")
    for label in sorted(set(y)):
        mask = y == label
        acc = np.mean(pred[mask] == label)
        if acc < 0.9:
            wrong = pred[mask][pred[mask] != label]
            values, counts = np.unique(wrong, return_counts=True)
            print(f"  {label}: {acc:.0%} (mostly mistaken for {values[np.argmax(counts)]})")


# ---------- letters ----------

def augment_letters(lm, rng, copies=2):
    out = [lm]
    for _ in range(copies):
        rotated = np.stack([rotate_xy(s, rng.uniform(-12, 12)) for s in lm])
        out.append(rotated + rng.normal(0, 0.004, lm.shape).astype(np.float32))
    return np.concatenate(out)


def train_letters():
    rng = np.random.default_rng(0)
    train, test = [], []
    for path in sorted(cfg.STATIC_DATA.glob("*/*.npy")):
        label, weight = path.stem, USER_WEIGHT if path.parent.name == "user" else 1
        if label not in cfg.LETTERS:
            continue
        lm = np.load(path)
        # Consecutive frames are near-duplicates, so hold out the last fifth
        # of each recording rather than a random sample.
        cut = int(len(lm) * 0.8)
        train.append((label, lm[:cut], weight))
        test.append((label, lm[cut:], weight))
    if not train:
        print("letters: no data under data/static - run `python collect.py letters`")
        return
    missing = sorted(set(cfg.LETTERS) - {label for label, _, _ in train})
    if missing:
        print(f"letters: no samples for {' '.join(missing)} - they cannot be recognised")

    def build(parts):
        x = [np.repeat(static_features(augment_letters(lm, rng)), w, axis=0) for _, lm, w in parts]
        y = [np.full(len(f), label) for f, (label, _, _) in zip(x, parts)]
        return np.concatenate(x), np.concatenate(y)

    x, y = build(train)
    x_test = np.concatenate([static_features(lm) for _, lm, _ in test])
    y_test = np.concatenate([np.full(len(lm), label) for label, lm, _ in test])
    report("letters", fit(x, y, (128, 64)), x_test, y_test)

    x_all, y_all = build(train + test)
    cfg.MODELS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": fit(x_all, y_all, (128, 64))}, cfg.LETTER_MODEL)
    print(f"letters: saved {cfg.LETTER_MODEL.name} ({len(set(y_all))} letters, {len(y_all)} samples)")


# ---------- signs ----------

def augment_clip(clip, rng, copies=CLIP_VARIANTS):
    """Feature vectors for a clip plus time-cropped, rotated and jittered variants."""
    t, hands, present = clip["t"], clip["hands"], clip["present"]
    out = [clip_features(t, hands, present)]
    for _ in range(copies):
        start, end = rng.uniform(0, 0.2), rng.uniform(0.8, 1.0)
        keep = (t >= start * t[-1]) & (t <= end * t[-1])
        moved = rotate_xy(hands[keep], rng.uniform(-10, 10))
        moved = moved + rng.normal(0, 0.003, moved.shape).astype(np.float32)
        out.append(clip_features(t[keep], moved, present[keep]))
    return out


def train_signs():
    rng = np.random.default_rng(0)
    clips = {}
    for folder in sorted(p for p in cfg.DYNAMIC_DATA.glob("*") if p.is_dir()):
        files = sorted(folder.glob("*.npz"))
        if len(files) >= MIN_CLIPS:
            clips[folder.name] = [dict(np.load(f)) for f in files]
        elif files:
            print(f"signs: skipping {folder.name} - {len(files)} clips, need at least {MIN_CLIPS}")
    if len(clips) < 2:
        print("signs: need at least two signs with data - run `python collect.py signs`")
        return
    if cfg.IDLE not in clips:
        print("signs: WARNING no idle clips recorded - random movement will be read as signs")

    def build(select):
        x, y = [], []
        for label, items in clips.items():
            for clip in select(items):
                feats = augment_clip(clip, rng)
                x += feats
                y += [label] * len(feats)
        return np.stack(x), np.array(y)

    # hold out every fifth clip, unaugmented
    x, y = build(lambda items: [c for i, c in enumerate(items) if i % 5 != 4])
    held = [(label, c) for label, items in clips.items() for i, c in enumerate(items) if i % 5 == 4]
    x_test = np.stack([clip_features(c["t"], c["hands"], c["present"]) for _, c in held])
    y_test = np.array([label for label, _ in held])
    report("signs", fit(x, y, (256, 128)), x_test, y_test)

    x_all, y_all = build(lambda items: items)
    cfg.MODELS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": fit(x_all, y_all, (256, 128))}, cfg.SIGN_MODEL)
    names = [n for n in clips if n != cfg.IDLE]
    print(f"signs: saved {cfg.SIGN_MODEL.name} ({len(names)} signs: {' '.join(names)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", nargs="?", choices=["letters", "signs"])
    what = parser.parse_args().what
    if what != "signs":
        train_letters()
    if what != "letters":
        train_signs()
