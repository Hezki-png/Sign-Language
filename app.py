"""Real-time sign language interpreter.

    python app.py [--camera 0]

Letters are typed by holding a fingerspelled handshape still; dynamic signs are
recognised from movement. Both run at once in AUTO mode.
"""
import argparse
import time

import cv2
import numpy as np

from signlang import config as cfg
from signlang import ui
from signlang.features import FrameBuffer, canonical, motion, pick_hand
from signlang.recognizer import LetterRecognizer, SignRecognizer, load_model
from signlang.text import Composer, load_words
from signlang.tracker import HandTracker

WINDOW = "Sign Language Interpreter"
MODES = ["AUTO", "SPELL", "SIGNS"]
HINT = "TAB mode   SPACE space   BACKSPACE delete   1-3 accept suggestion   C clear   ESC quit"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    letter_bundle, sign_bundle = load_model(cfg.LETTER_MODEL), load_model(cfg.SIGN_MODEL)
    letters = LetterRecognizer(letter_bundle) if letter_bundle else None
    signs = SignRecognizer(sign_bundle) if sign_bundle else None
    composer = Composer(load_words())
    tracker = HandTracker()
    cap = ui.open_camera(args.camera)
    buffer = FrameBuffer(cfg.CLIP_SECONDS)

    mode = 0
    last_hand_t = time.perf_counter()
    letters_blocked_until = 0.0
    flash, flash_until = "", 0.0
    fps, prev_t = 0.0, time.perf_counter()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while True:
        frame = ui.read_mirrored(cap)
        if frame is None:
            print("Camera stopped delivering frames.")
            break
        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(t - prev_t, 1e-6)
        prev_t = t

        hands = tracker.process(frame)
        ui.warn_if_blank(frame)
        buffer.push(t, hands)
        ui.draw_hands(frame, hands)

        # --- dynamic signs ---
        sign_label, sign_conf = None, 0.0
        if signs and MODES[mode] != "SPELL":
            sign_label, sign_conf, fired = signs.update(buffer)
            if fired:
                if len(fired) == 1:  # J and Z are trained as motion signs
                    composer.add_letter(fired)
                else:
                    composer.add_word(fired.replace("_", " "))
                flash, flash_until = fired.replace("_", " "), t + 1.2
                letters_blocked_until = t + cfg.LETTER_BLOCK_AFTER_SIGN
                if letters:
                    letters.reset()

        # --- fingerspelling ---
        letter_label, letter_conf, progress = None, 0.0, 0.0
        hand = pick_hand(hands)
        if hand is not None:
            last_hand_t = t
            if letters and MODES[mode] != "SIGNS" and t >= letters_blocked_until:
                still = motion(*buffer.arrays(), cfg.STILL_WINDOW) < cfg.STILL_MAX_MOTION
                letter_label, letter_conf, progress, typed = letters.update(canonical(hand), t, still)
                if typed:
                    composer.add_letter(typed)
        else:
            if letters:
                letters.reset()
            if t - last_hand_t > cfg.WORD_GAP:
                composer.end_word()

        # --- overlay ---
        ui.put(frame, f"{MODES[mode]}   {fps:4.1f} fps", (16, 30), 0.6, ui.GREY)
        if letter_label and letter_conf >= cfg.LETTER_MIN_CONF:
            ui.put(frame, letter_label, (24, 130), 3.0, ui.WHITE, 6)
            ui.put(frame, f"{letter_conf:.0%}", (120, 100), 0.7, ui.GREY)
            ui.bar(frame, 24, 146, 150, 10, progress)
        if sign_label:
            ui.put(frame, f"{sign_label.replace('_', ' ')}?  {sign_conf:.0%}", (24, 190), 0.8, ui.AMBER)
        if t < flash_until:
            ui.put(frame, flash, (24, frame.shape[0] - 30), 1.6, ui.GREEN, 4)
        y = 60
        for missing, script in ((letters, "letters"), (signs, "signs")):
            if missing is None:
                ui.put(frame, f"No {script} model yet - see README (collect.py {script}, then train.py)",
                       (16, y), 0.55, ui.AMBER, 1)
                y += 24

        panel = ui.text_panel(frame.shape[1], composer.text, composer.suggestions(), HINT)
        cv2.imshow(WINDOW, np.vstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")) or ui.window_closed(WINDOW):
            break
        elif key == 9:
            mode = (mode + 1) % len(MODES)
        elif key == 32:
            composer.end_word()
        elif key == 8:
            composer.backspace()
        elif key == ord("c"):
            composer.clear()
        elif key in (ord("1"), ord("2"), ord("3")):
            composer.accept(key - ord("1"))

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if composer.text:
        print(composer.text)


if __name__ == "__main__":
    main()
