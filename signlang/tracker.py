"""Thin wrapper around MediaPipe's HandLandmarker."""
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .config import HAND_MODEL


@dataclass
class Hand:
    label: str        # "Right" or "Left"
    lm: np.ndarray    # (21, 3) landmarks, all axes in units of image height
    pts: np.ndarray   # (21, 2) landmarks normalised to [0, 1], for drawing


class HandTracker:
    """Detects up to two hands.

    MediaPipe's handedness assumes a mirrored (selfie) image, so feed it frames
    that are already flipped if "Right" should mean the signer's right hand.
    """

    def __init__(self, video=True, num_hands=2):
        if not HAND_MODEL.exists():
            raise FileNotFoundError(f"Missing {HAND_MODEL} - see README for the download link.")
        self.video = video
        self._last_ms = -1
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands,
        ))

    def process(self, bgr):
        h, w = bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.video:
            # timestamps must be strictly increasing
            self._last_ms = max(self._last_ms + 1, int(time.perf_counter() * 1000))
            result = self._landmarker.detect_for_video(image, self._last_ms)
        else:
            result = self._landmarker.detect(image)

        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            xyz = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            hands.append(Hand(label=handedness[0].category_name,
                              lm=xyz * np.float32([w / h, 1, w / h]),
                              pts=xyz[:, :2]))
        return hands

    def close(self):
        self._landmarker.close()
