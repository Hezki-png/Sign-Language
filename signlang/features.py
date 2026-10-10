"""Turn raw landmarks into classifier features.

Raw landmarks are what gets saved to disk, so the features can change without
re-recording any data.
"""
from collections import deque

import numpy as np

from .config import CLIP_FRAMES

SLOTS = {"Right": 0, "Left": 1}
CLIP_FEATURES = CLIP_FRAMES * 2 * 68


def canonical(hand):
    """Landmarks of a hand as if it were a right hand (left hands are mirrored)."""
    lm = hand.lm.copy()
    if hand.label == "Left":
        lm[:, 0] = -lm[:, 0]
    return lm


def pick_hand(hands):
    """The hand used for fingerspelling: the right one if both are visible."""
    return min(hands, key=lambda h: SLOTS.get(h.label, 2)) if hands else None


def static_features(lm):
    """(..., 21, 3) landmarks -> (..., 63) wrist-centred, palm-scaled coordinates.

    Rotation is deliberately kept: orientation distinguishes letters such as
    G/Q, K/P and H/U.
    """
    lm = np.asarray(lm, dtype=np.float32)
    p = lm - lm[..., :1, :]
    scale = np.linalg.norm(p[..., 9, :], axis=-1)[..., None, None]
    return (p / np.maximum(scale, 1e-6)).reshape(*lm.shape[:-2], 63)


def rotate_xy(lm, degrees):
    """Rotate (..., 21, 3) landmarks in the image plane."""
    a = np.deg2rad(degrees)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)
    return lm @ rot.T


def pack_hands(hands):
    """List of Hand -> ((2, 21, 3) landmarks, (2,) present) with fixed Right/Left slots."""
    lm = np.zeros((2, 21, 3), dtype=np.float32)
    present = np.zeros(2, dtype=bool)
    for hand in hands:
        slot = SLOTS.get(hand.label)
        if slot is not None and not present[slot]:
            lm[slot], present[slot] = hand.lm, True
    return lm, present


def _palm(hands):
    return np.linalg.norm(hands[..., 9, :] - hands[..., 0, :], axis=-1)


def clip_features(t, hands, present, n=CLIP_FRAMES):
    """A clip (t (F,), hands (F, 2, 21, 3), present (F, 2)) -> flat feature vector.

    Per frame and hand: hand shape, wrist position relative to the clip's
    centre, wrist velocity and a presence flag. Positions are in palm lengths,
    so distance from the camera does not matter.
    """
    if len(t) == 0 or not present.any():
        return np.zeros(CLIP_FEATURES, dtype=np.float32)
    targets = np.linspace(t[0], t[-1], n)
    idx = np.clip(np.searchsorted(t, targets), 0, len(t) - 1)
    hands, present = hands[idx], present[idx]
    if not present.any():
        return np.zeros(CLIP_FEATURES, dtype=np.float32)

    mask = present[..., None].astype(np.float32)
    wrist = hands[:, :, 0, :2]
    scale = max(float(np.median(_palm(hands)[present])), 1e-6)
    pos = (wrist - wrist[present].mean(axis=0)) / scale * mask
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * (present[1:] & present[:-1])[..., None]
    shape = static_features(hands) * mask
    return np.concatenate([shape, pos, vel, mask], axis=-1).reshape(-1).astype(np.float32)


def motion(t, hands, present, seconds):
    """How far the busiest hand's landmarks travelled over the last `seconds`, in palm lengths."""
    if len(t) < 2:
        return 0.0
    recent = t >= t[-1] - seconds
    best = 0.0
    for slot in range(2):
        ok = recent & present[:, slot]
        if ok.sum() < 2:
            continue
        xy = hands[ok, slot, :, :2]
        travel = np.linalg.norm(np.diff(xy, axis=0), axis=-1).mean(axis=1).sum()
        best = max(best, float(travel / max(np.median(_palm(hands[ok, slot])), 1e-6)))
    return best


class FrameBuffer:
    """Sliding window of the most recent tracked frames."""

    def __init__(self, seconds):
        self.seconds = seconds
        self._frames = deque()

    def push(self, t, hands):
        self._frames.append((t, *pack_hands(hands)))
        while t - self._frames[0][0] > self.seconds:
            self._frames.popleft()

    def clear(self):
        self._frames.clear()

    @property
    def span(self):
        return self._frames[-1][0] - self._frames[0][0] if self._frames else 0.0

    def arrays(self):
        if not self._frames:
            return np.zeros(0), np.zeros((0, 2, 21, 3), np.float32), np.zeros((0, 2), bool)
        t, hands, present = zip(*self._frames)
        return np.array(t), np.stack(hands), np.stack(present)
