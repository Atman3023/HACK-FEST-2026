# SignBridge — CLAUDE.md

Hospital-reception communication assistant for **Indian Sign Language (ISL)**, built for a hackathon.
**Deadline: 9:30pm (2026-11-04).** Keep everything simple and demo-quality. Prefer working over clever.

## Hard rules
- Vocabulary is **strictly the ISL letters A, B, C**. Nothing else.
- **Never claim general ISL translation** — in UI text, README, or pitch. Say "recognises ISL letters A, B and C".
- Always show **confidence bars** for every letter on every result.
- Everything runs **offline/local** (no cloud APIs), including text-to-speech.

## Environment
- **Python 3.12** is required — MediaPipe supports 3.10–3.12 only. The machine's default is 3.14 (3.13 also installed); neither works.
  - Install: `winget install Python.Python.3.12`
  - Venv: `py -3.12 -m venv .venv` then `.venv\Scripts\activate` and `pip install -r requirements.txt`
- Pin `mediapipe==0.10.14` and use the legacy `mp.solutions.hands` API (no `.task` model file to download).
- Run: `streamlit run app.py`

## Data
- Raw images live in `data/raw/`. **Assumed layout: `data/raw/<LETTER>/*.jpg|png`** (one folder per letter A/B/C).
  _TODO: confirm once the images are added — update this section with the real layout and per-letter counts._
- `scripts/build_library.py` turns raw images into the sign library `data/library/landmarks.npz`
  (landmark vectors + labels + source filename). Images where MediaPipe finds no hand are skipped and logged.
- ISL reference images shown to users: `assets/isl/{A,B,C}.png` (picked from data/raw).

## App (Streamlit)
Home screen with two services:

### 1. Sign → Text/Speech
1. User takes a webcam snapshot (`st.camera_input`).
2. MediaPipe Hands detects **up to 2 hands** (`max_num_hands=2`; ISL letters can be two-handed).
3. Extract **normalized landmarks**: per hand, subtract wrist, scale by max distance from wrist;
   order hands deterministically (left-to-right by wrist x); one-hand signs get the second slot zero-padded
   plus a hand-count feature, so 1-hand vs 2-hand never match each other cheaply.
4. **Comparator engine** brute-force compares the vector against **every sample** in the library
   (Euclidean distance). Per letter take the best (min) distance, convert to confidence with
   softmax over `-d / T` → a confidence score for each of A, B, C.
5. Feed into the decision logic.

### 2. Text/Drawing → Sign
- **Draw**: user draws a letter on a canvas (`streamlit-drawable-canvas`). A handwriting recognizer
  (template matching: crop → resize to 28×28 → compare against rendered/drawn A/B/C templates) gives a confidence per letter.
- **Type**: user types a letter; A/B/C (case-insensitive) → 100% confidence; anything else → "only A, B, C supported".
- Shows the matching ISL reference image.

### Decision logic (shared, `signbridge/decision.py`)
| Top confidence | Behaviour |
|---|---|
| `< 60%` | Ask the user to **retry** |
| `60% – <85%` | Show **top 2 candidates** for the user to confirm |
| `≥ 85%` | **Final answer**: letter as text + ISL image + optional offline TTS (`pyttsx3`) |

Thresholds live in `signbridge/config.py` (`RETRY_BELOW = 0.60`, `FINAL_AT = 0.85`).

## Held-out accuracy (must report)
- `scripts/evaluate.py`: stratified train/test split of the library (fixed seed), test samples are
  compared only against train samples. Report overall accuracy, per-letter accuracy, confusion matrix,
  and how often each decision band (retry / confirm / final) fires. Write to `reports/accuracy.md`.
- Leave-one-out is fine too if the dataset is tiny. Never evaluate a sample against itself.
- Show the reported number in the app (e.g. on the home screen / about box).

## File structure
```
app.py                      # Streamlit home + the two services
requirements.txt
signbridge/
  config.py                 # letters, thresholds, paths
  landmarks.py              # MediaPipe detection + normalization
  comparator.py             # brute-force library compare -> per-letter confidence
  handwriting.py            # canvas letter recognizer
  decision.py               # retry / confirm / final
  tts.py                    # pyttsx3 wrapper (fails gracefully)
  ui.py                     # confidence bars, shared widgets
scripts/
  build_library.py          # data/raw -> data/library/landmarks.npz
  evaluate.py               # held-out accuracy -> reports/accuracy.md
data/raw/                   # provided images
data/library/               # generated
assets/isl/                 # A.png, B.png, C.png reference images
reports/                    # accuracy report
```

## Build order (time-boxed)
1. Python 3.12 venv + requirements; confirm MediaPipe imports.
2. `landmarks.py` + `build_library.py`; check detection rate on data/raw.
3. `comparator.py` + `evaluate.py` → first accuracy number.
4. `decision.py` + `ui.py` + Sign → Text service in `app.py`.
5. Text/Drawing → Sign service (`handwriting.py`).
6. TTS, polish, README, final accuracy report.
