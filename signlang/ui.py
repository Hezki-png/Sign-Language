"""Camera access and OpenCV drawing helpers."""
import sys

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, AMBER, DARK = (255, 255, 255), (170, 170, 170), (90, 220, 90), (60, 190, 255), (28, 28, 28)

_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
          (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def open_camera(index=0, width=1280, height=720):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Try --camera 1, or close other apps using it.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_mirrored(cap):
    """Next frame flipped like a mirror, or None if the camera stopped."""
    ok, frame = cap.read()
    return cv2.flip(frame, 1) if ok else None


def warn_if_blank(frame):
    """Explain a featureless picture (closed shutter, camera privacy mode) instead of showing a dead window."""
    if frame[::16, ::16].std() < 2:
        put(frame, "The camera is sending a blank picture.", (40, frame.shape[0] // 2 - 20), 1.0, AMBER)
        put(frame, "Open its privacy shutter or turn off camera privacy mode (camera key / Lenovo Vantage).",
            (40, frame.shape[0] // 2 + 20), 0.6, WHITE, 1)


def window_closed(name):
    return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1


def put(img, s, org, scale=0.7, color=WHITE, thick=2):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color=GREEN):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * min(max(frac, 0.0), 1.0)), y + h), color, -1)


def draw_hands(img, hands):
    h, w = img.shape[:2]
    for hand in hands:
        pts = (hand.pts * (w, h)).astype(int)
        for a, b in _BONES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), GREEN, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, WHITE, -1, cv2.LINE_AA)


def _tail_that_fits(s, scale, thick, max_width):
    while s and cv2.getTextSize(s, FONT, scale, thick)[0][0] > max_width:
        s = s[1:]
    return s


def text_panel(width, text, suggestions=(), hint="", height=150):
    """The strip under the video: suggestions, the running text and a key hint."""
    panel = np.full((height, width, 3), DARK, dtype=np.uint8)
    x = 20
    for i, word in enumerate(suggestions):
        label = f"{i + 1}: {word}"
        cv2.putText(panel, label, (x, 34), FONT, 0.7, AMBER, 2, cv2.LINE_AA)
        x += cv2.getTextSize(label, FONT, 0.7, 2)[0][0] + 36
    shown = _tail_that_fits(text + "_", 1.3, 3, width - 40)
    cv2.putText(panel, shown, (20, 92), FONT, 1.3, WHITE, 3, cv2.LINE_AA)
    cv2.putText(panel, hint, (20, height - 14), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    return panel
