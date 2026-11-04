# SignBridge — CLAUDE.md

Hospital-reception communication assistant for **Indian Sign Language (ISL)**, built for a hackathon.
**Deadline: 9:30pm (2026-11-04).** Keep everything simple and demo-quality. Prefer working over clever.

## Hard rules
- Vocabulary is **strictly the ISL letters A, B, C**. Nothing else.
- **Never claim general ISL translation** — in UI text, README, or pitch. Say "recognises ISL letters A, B and C".
- Always show **confidence bars** for every letter on every result.
- Everything runs **offline/local** (no cloud APIs), including text-to-speech.
- **Calibration is tuned on training data only — never on test data.**

## Environment
- **Python 3.12** in `.venv` (MediaPipe supports 3.10–3.12; the machine default is 3.14, which won't work).
  - Setup: `py -3.12 -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt`
- Pins that matter (see comments in `requirements.txt`):
  - `mediapipe==0.10.14`, legacy `mp.solutions.hands` API (no `.task` model download).
  - `numpy==1.26.4` — mediapipe 0.10.14 needs numpy < 2.
  - `pyarrow==17.0.0` — newer pyarrow needs numpy 2 and breaks the drawing canvas.
  - `opencv-contrib-python` only (comes with mediapipe); never also install `opencv-python`.
  - `streamlit==1.39.0` — `streamlit-drawable-canvas` breaks on >= 1.40.
- Run modules from the repo root: `python -m src.extract` (not `python src/extract.py`).

## Pipeline
```
python -m src.extract    # data/raw -> data/library/landmarks.npz
python -m src.train      # split + calibrate on train -> models/*.npz
python -m src.evaluate   # held-out accuracy -> reports/accuracy.md + .json
streamlit run app.py
```

## Data
- `data/raw/` — provided sign images. **Assumed layout: `data/raw/<LETTER>/*.jpg|png`**; a filename
  like `A_01.jpg` also works. _TODO: confirm once the images are added — record real layout and per-letter counts._
- `data/handwriting_test/<LETTER>/*.png` — small set of real handwritten A/B/C drawings (black on white)
  used only to report handwriting accuracy. Never used to build templates or calibrate.
- `assets/isl/{A,B,C}.png` — **one clean reference image per letter**, shown to users.
- Generated: `data/library/landmarks.npz`, `models/sign_model.npz`, `models/handwriting_model.npz`, `reports/`.

## App (Streamlit, `app.py`)
Home screen with two services, plus a held-out accuracy box and the A/B/C-only scope note.

### 1. Sign → Text/Speech
1. Webcam snapshot (`st.camera_input`).
2. MediaPipe Hands, **up to 2 hands** (`max_num_hands=2`; ISL letters can be two-handed).
3. **Normalized landmarks** (`src/extract.py`): pixel coords (isotropic), hands ordered left→right,
   centred on the wrist (or wrist midpoint for two hands), scaled so the farthest point = 1 — keeps
   the two hands' relative pose. Second-hand slot zero-padded + a hand-count feature so 1-hand and
   2-hand signs don't match. A mirrored variant is also computed so left-handed signers match.
4. **Comparator** (`src/comparator.py`) brute-force computes the distance to **every library sample**
   (min over original/mirrored), then per letter **averages the 3 nearest samples** (k = 3).
   Confidence = softmax(−d / T). If even the best letter is farther than `d_max`, the sign is
   flagged "not A/B/C" → retry.
5. **Calibration** (`calibrate()`): T (minimum NLL) and `d_max` (1.5 × p99 of true-class distance)
   are tuned with leave-one-out **on the training split only**.

### 2. Text/Drawing → Sign
- **Draw** (`streamlit-drawable-canvas`): `src/handwriting.py` crops → pads square → 32×32 → blur →
  unit-norm vector, then uses the same comparator against **templates rendered from 16 Windows fonts
  × 4 stroke thicknesses × upper/lower case** (384 templates). T is calibrated leave-one-font-out on
  the templates (the handwriting "training data").
- **Type**: A/B/C (case-insensitive) → 100% confidence; anything else → "only A, B and C are supported".
- Shows the matching ISL reference image from `assets/isl/`.

### Decision logic (shared, `src/decision.py`)
| Top confidence | Behaviour |
|---|---|
| `< 60%` (or out of range) | Ask the user to **retry** |
| `60% – <85%` | Show **top 2 candidates** for the user to confirm |
| `≥ 85%` | **Final answer**: letter as text + ISL image + optional offline TTS (`pyttsx3`, run in a subprocess) |

Thresholds live in `src/config.py` (`RETRY_BELOW = 0.60`, `FINAL_AT = 0.85`, `K_NEAREST = 3`).

## Held-out accuracy (must report)
- `src/train.py`: stratified split (`TEST_SIZE = 0.25`, `SEED = 42`), saves train/test indices.
- `src/evaluate.py`: test samples are compared **only against train samples** using the train-calibrated
  T. Reports top-1 accuracy, per-letter accuracy, confusion matrix, decision-band counts, precision of the
  "final" band and top-2 hit rate of the "confirm" band. Same report for `data/handwriting_test/`.
- The app library uses all samples, but T/`d_max` still come from the training split only.
- Caveat to mention honestly: if data/raw images are near-duplicate frames of the same person/session,
  a random split inflates accuracy.

## File structure
```
app.py                      # Streamlit home + the two services
requirements.txt
src/
  config.py                 # letters, thresholds, paths, label-from-path
  extract.py                # MediaPipe detection + normalization; builds sign library
  comparator.py             # brute-force k=3 comparator + train-only calibration
  handwriting.py            # canvas preprocessing + multi-font/thickness templates
  decision.py               # retry / confirm / final
  tts.py                    # offline TTS (pyttsx3 in a subprocess)
  train.py                  # split, calibrate, save models
  evaluate.py               # held-out accuracy report
data/raw/                   # provided sign images
data/handwriting_test/      # small real handwriting test set
data/library/               # generated
assets/isl/                 # A.png, B.png, C.png (one clean reference each)
models/  reports/           # generated
```

## Status / next steps
1. Done: venv, requirements, all modules, app pages smoke-tested on synthetic data.
2. Add images to `data/raw/`, run extract → check detection rate per letter.
3. train → evaluate → first real accuracy number.
4. Pick `assets/isl/{A,B,C}.png`; collect ~10 drawings per letter for `data/handwriting_test/`.
5. Live webcam test, polish, README, final accuracy report.
