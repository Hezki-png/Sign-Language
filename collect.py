"""Record your own training samples.

    python collect.py letters              press a letter key, hold the handshape
    python collect.py signs [NAME ...]     pick a sign with [ and ], SPACE records one clip

Run train.py afterwards.
"""
import argparse
import time

import cv2
import numpy as np

from signlang import config as cfg
from signlang import ui
from signlang.features import canonical, pack_hands, pick_hand
from signlang.tracker import HandTracker

WINDOW = "Collect samples"
LETTER_SAMPLES = 60   # frames saved per key press
COUNTDOWN = 1.5       # seconds between the key press and the recording


def frames(cap, tracker):
    """Yields (frame, hands) with the skeleton drawn, until the camera stops or the window is closed."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while True:
        frame = ui.read_mirrored(cap)
        if frame is None:
            print("Camera stopped delivering frames.")
            return
        hands = tracker.process(frame)
        ui.warn_if_blank(frame)
        ui.draw_hands(frame, hands)
        yield frame, hands
        if ui.window_closed(WINDOW):
            return


def show(frame, text, hint):
    cv2.imshow(WINDOW, np.vstack([frame, ui.text_panel(frame.shape[1], text, hint=hint)]))
    return cv2.waitKey(1) & 0xFF


def collect_letters(cap, tracker):
    folder = cfg.STATIC_DATA / "user"
    folder.mkdir(parents=True, exist_ok=True)
    counts = {c: len(np.load(folder / f"{c}.npy")) if (folder / f"{c}.npy").exists() else 0 for c in cfg.LETTERS}
    target, start_t, samples = None, 0.0, []

    for frame, hands in frames(cap, tracker):
        t = time.perf_counter()
        hand = pick_hand(hands)
        if target and t >= start_t:
            if hand is not None:
                samples.append(canonical(hand))
            ui.put(frame, f"Recording {target}  {len(samples)}/{LETTER_SAMPLES}", (20, 50), 1.0, ui.GREEN)
            ui.put(frame, "tilt and move your hand slightly", (20, 84), 0.6, ui.GREY, 1)
            if len(samples) >= LETTER_SAMPLES:
                path = folder / f"{target}.npy"
                data = np.stack(samples)
                if path.exists():
                    data = np.concatenate([np.load(path), data])
                np.save(path, data)
                counts[target] = len(data)
                target, samples = None, []
        elif target:
            ui.put(frame, f"Get ready: {target}  {start_t - t:.1f}", (20, 50), 1.0, ui.AMBER)

        todo = "".join(c for c in cfg.LETTERS if counts[c] == 0)
        status = f"TO DO: {todo}" if todo else "ALL LETTERS RECORDED"
        key = show(frame, status, "press a letter key to record it (J and Z are signs, not letters)   ESC quit")
        if key == 27:
            break
        if target is None and chr(key).upper() in cfg.LETTERS:
            target, start_t, samples = chr(key).upper(), t + COUNTDOWN, []


def collect_signs(cap, tracker, names):
    names = [n.upper() for n in names] + [cfg.IDLE]
    current, start_t, clip, saved = 0, None, [], []

    def count(name):
        return len(list((cfg.DYNAMIC_DATA / name).glob("*.npz")))

    for frame, hands in frames(cap, tracker):
        t = time.perf_counter()
        name = names[current]
        if start_t is not None and t >= start_t:
            clip.append((t, *pack_hands(hands)))
            elapsed = t - start_t
            ui.put(frame, f"Recording {name}", (20, 50), 1.0, ui.GREEN)
            ui.bar(frame, 20, 66, 300, 12, elapsed / cfg.CLIP_SECONDS)
            if elapsed >= cfg.CLIP_SECONDS:
                ts, lm, present = zip(*clip)
                folder = cfg.DYNAMIC_DATA / name
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{time.time_ns()}.npz"
                np.savez_compressed(path, t=np.array(ts) - ts[0], hands=np.stack(lm), present=np.stack(present))
                saved.append(path)
                start_t, clip = None, []
        elif start_t is not None:
            ui.put(frame, f"Get ready: {name}  {start_t - t:.1f}", (20, 50), 1.0, ui.AMBER)

        label = "IDLE (rest, fidget, fingerspell - anything that is not a sign)" if name == cfg.IDLE else name
        key = show(frame, f"{label}  x{count(name)}",
                   "SPACE record a clip   [ ] previous / next sign   BACKSPACE undo last clip   ESC quit")
        if key == 27:
            break
        if start_t is not None:
            continue
        if key == 32:
            start_t, clip = t + COUNTDOWN, []
        elif key == ord("["):
            current = (current - 1) % len(names)
        elif key == ord("]"):
            current = (current + 1) % len(names)
        elif key == 8 and saved:
            saved.pop().unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["letters", "signs"])
    parser.add_argument("names", nargs="*", help="sign names (default: the built-in starter vocabulary)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    tracker = HandTracker()
    cap = ui.open_camera(args.camera)
    if args.what == "letters":
        collect_letters(cap, tracker)
    else:
        collect_signs(cap, tracker, args.names or cfg.DEFAULT_SIGNS)
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
