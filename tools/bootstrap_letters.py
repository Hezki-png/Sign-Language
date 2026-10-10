"""Build starter letter data from a labelled image dataset in Hugging Face parquet format.

    pip install pyarrow
    python tools/bootstrap_letters.py path/to/train.parquet

Expects an `image` column ({bytes, path}) and an integer `label` column where
0..25 map to A..Z. Runs the hand tracker over every image and stores the
landmarks under data/static/bootstrap/.
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signlang import config as cfg  # noqa: E402
from signlang.features import canonical  # noqa: E402
from signlang.tracker import HandTracker  # noqa: E402


def main(parquet_path):
    tracker = HandTracker(video=False, num_hands=1)
    samples, seen, missed = defaultdict(list), 0, 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=256):
        for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            letter = chr(ord("A") + label)
            if letter not in cfg.LETTERS:
                continue
            seen += 1
            bgr = cv2.imdecode(np.frombuffer(image["bytes"], np.uint8), cv2.IMREAD_COLOR)
            hands = tracker.process(bgr) if bgr is not None else []
            if hands:
                samples[letter].append(canonical(hands[0]))
            else:
                missed += 1
        print(f"\r{seen} images, {missed} without a detectable hand", end="", flush=True)
    print()

    folder = cfg.STATIC_DATA / "bootstrap"
    folder.mkdir(parents=True, exist_ok=True)
    for letter, items in sorted(samples.items()):
        np.save(folder / f"{letter}.npy", np.stack(items))
        print(f"  {letter}: {len(items)}")


if __name__ == "__main__":
    main(sys.argv[1])
