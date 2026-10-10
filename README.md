# Real-time ASL interpreter

Watches your hands through a webcam, recognises ASL fingerspelling and a small
vocabulary of moving signs, and writes the result as running text.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py            # add --camera 1 for a second webcam
```

`models/hand_landmarker.task` is Google's hand-tracking model. If it is missing, download it from
<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>.

### Using the app

- **Letters:** hold a handshape still for about half a second and it is typed. Keep holding to type it again
  (double letters). The bar under the big letter shows the hold progress.
- **Words:** take your hand out of view for 1.5 s, or press SPACE, to end a word.
- **Suggestions:** while spelling, the three most likely words appear above the text; press 1, 2 or 3 to accept one.
- **Signs:** just sign. A recognised sign is added as a whole word.

| Key | Action |
| --- | --- |
| TAB | cycle AUTO / SPELL (letters only) / SIGNS (signs only) |
| SPACE | end the current word |
| BACKSPACE | delete a letter, or the last word |
| 1 2 3 | accept a suggestion |
| C | clear the text |
| ESC or Q | quit |

## Teaching it

The app ships with a starter **letter** model. There is no sign model until you record one.

### Letters (2 minutes per pass, strongly recommended)

```powershell
.\.venv\Scripts\python collect.py letters
.\.venv\Scripts\python train.py letters
```

Press a letter key, get the handshape ready during the countdown, then tilt and move your hand slightly while 60
frames are captured. Your own samples are weighted 3x over the starter data, so recording the letters the app gets
wrong for you is the fastest way to fix them.

### Signs

```powershell
.\.venv\Scripts\python collect.py signs                    # starter vocabulary
.\.venv\Scripts\python collect.py signs HELLO YES NO       # or your own list
.\.venv\Scripts\python train.py signs
```

`[` and `]` choose the sign, SPACE records one 1.5-second clip after a short countdown, BACKSPACE deletes the clip you
just recorded. Record **at least 15 clips per sign** (5 is the hard minimum), and the same number for the `IDLE`
entry at the end of the list: resting hands, scratching your face, fingerspelling, moving into position. Without idle
clips every movement is read as some sign.

J and Z need motion, so they are in the sign list; a one-letter sign is typed as a letter.

`train.py` prints accuracy on held-out samples and lists the classes it confuses.

## How it works

```
webcam -> MediaPipe HandLandmarker (21 3D points per hand)
       -> letters: wrist-centred, palm-scaled points      -> MLP -> hold-to-type
       -> signs:   1.5 s sliding window, 20 frames of
                   hand shape + wrist path + velocity     -> MLP -> fire after 4 agreeing frames
       -> Composer: letters + words -> sentence, prefix autocomplete
```

- Left hands are mirrored, so letters work with either hand.
- Letters are only typed while the hand is still; signs are only considered while it is moving. That is what lets
  both run together in AUTO mode.
- Raw landmarks are stored under `data/`, not features, so `signlang/features.py` can change without re-recording.
- Every threshold (hold time, confidence, motion gates) is in `signlang/config.py`.

| Path | Purpose |
| --- | --- |
| `app.py` | the interpreter |
| `collect.py` | record letter samples and sign clips |
| `train.py` | train `models/letters.joblib` and `models/signs.joblib` |
| `signlang/` | tracker, features, recognisers, text composer, drawing |
| `tools/bootstrap_letters.py` | rebuild the starter letter data from an image dataset |
| `data/words.txt` | word-frequency list for autocomplete |

## Known limits

- The starter letter model was trained on the public image set
  [Marxulia/asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03)
  (93.9% held-out accuracy). It is weakest on P, Q, V, R and N, and on the fist-shaped group A/E/M/N/S/T in general.
  That dataset states no licence, so check before redistributing `data/static/bootstrap/` or the model built from it.
- Only hands are tracked. Signs that differ only by where they are made on the body (chin vs. forehead) or by facial
  expression look identical to it.
- Signs are recognised as isolated gestures in a fixed 1.5 s window. Very slow or long signs will not fit.
- A sign model trained on one person works best for that person.
- `data/words.txt` is from [google-10000-english](https://github.com/first20hours/google-10000-english).
