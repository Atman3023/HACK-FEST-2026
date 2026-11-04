# SignBridge — CLAUDE.md

Hospital-reception communication assistant for **Indian Sign Language (ISL)**, built for a hackathon.
**Deadline: 9:30pm (2026-11-04).** Keep everything simple and demo-quality. Prefer working over clever.

## Hard rules
- Vocabulary is **exactly 6 signs**: letters **A, B, C** and actions **BLOOD_PRESSURE, FEVER, STOMACH_ACHE**.
  `NONE` is a rejection class ("not a known sign"), not a sign.
- **Never claim general ISL translation** — in UI text, README, or pitch. Say "recognises 6 ISL signs".
- Actions are **requests for the receptionist to route the visitor** (e.g. "Visitor indicates: fever"),
  never a diagnosis or symptom assessment. Not a replacement for an interpreter.
- Always show **confidence bars** for every class on every result.
- Everything runs **offline/local** (no cloud APIs), including text-to-speech.
- **Calibration is tuned on training data only — never on test data.**
- **Actions come after the letter pipeline works end to end on real data.**

## Environment
- **Python 3.12** in `.venv` (MediaPipe supports 3.10–3.12; the machine default is 3.14, which won't work).
  - Setup: `py -3.12 -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt`
- Pins that matter (see comments in `requirements.txt`):
  - `mediapipe==0.10.14`, legacy `mp.solutions` API. **Holistic verified working** on this pin:
    use `model_complexity=1` (bundled; complexity 0 downloads a model at first use). ~40 ms/frame on this laptop.
  - `numpy==1.26.4` — mediapipe 0.10.14 needs numpy < 2.
  - `pyarrow==17.0.0` — newer pyarrow needs numpy 2 and breaks the drawing canvas.
  - `opencv-contrib-python` only (comes with mediapipe); never also install `opencv-python`.
  - `streamlit==1.39.0` — `streamlit-drawable-canvas` breaks on >= 1.40.
- No new dependencies for actions: DTW is plain numpy, fuzzy matching is `difflib`. **No streamlit-webrtc.**
- Run modules from the repo root: `python -m src.extract` (not `python src/extract.py`).

## Pipeline
```
python -m src.extract           # letters: data/raw -> data/library/landmarks.npz
python -m src.extract_actions   # actions: data/actions -> data/library/actions.npz
python -m src.train             # split/calibrate on train -> models/*.npz (actions skipped if no data)
python -m src.evaluate          # held-out accuracy -> reports/accuracy.md + .json
streamlit run app.py
```

## Data
- `data/raw/` — letter images. **Assumed layout: `data/raw/<LETTER>/*.jpg|png`** (or `A_01.jpg`).
  _TODO: confirm once added — record real layout and per-letter counts._
- `data/actions/<CLASS>/<signer>_<n>.mp4` — action clips, CLASS ∈ {BLOOD_PRESSURE, FEVER, STOMACH_ACHE, NONE}.
  Signer = filename stem before the last `_`. `NONE/` holds non-sign movements (waving, scratching head,
  adjusting glasses, idle). Target: ≥ 3 signers × ~5 clips per class.
- `data/handwriting_test/<LETTER>/*.png` — real handwritten drawings, used only to report handwriting accuracy.
- `assets/isl/{A,B,C}.png` — one clean reference image per letter.
- `assets/isl/actions/{BLOOD_PRESSURE,FEVER,STOMACH_ACHE}.mp4` — our own reference clips.
  **Must be H.264** so the browser can play them (`st.video`); OpenCV-written `mp4v` files won't play.
- Generated: `data/library/*.npz`, `models/*.npz`, `reports/`.

## App (Streamlit, `app.py`)
Home screen with two services, a held-out accuracy box and the 6-sign scope note.

### Service 1: Sign → Text/Speech — two sub-modes
**a) Spell a letter (snapshot)** — unchanged:
`st.camera_input` → MediaPipe Hands (≤ 2 hands) → normalized landmarks (wrist-centred, scaled, mirrored
variant) → brute-force comparator, mean of 3 nearest per letter → softmax(−d/T) → decision.

**b) Show an action (3-second clip)**
1. **Capture** (`src/capture.py`): OpenCV `VideoCapture(0, CAP_DSHOW)` on the local webcam; live frames
   pushed to an `st.image` placeholder; 3-2-1 countdown overlay, then record **3 s by wall-clock time**
   (the camera reports fps = 0), then release the camera.
   Risk: browser (`camera_input`) and OpenCV may not share the webcam on Windows — never render both in
   the same run; test early. Fallback: switch letter snapshot to OpenCV too.
2. **Holistic** (`src/action_features.py`): MediaPipe Holistic on each recorded frame (`model_complexity=1`).
3. **Per-frame features**, all normalized by **shoulder width**:
   - references: nose (pose 0), chin (face mesh 152; fallback: mouth centre extrapolated from nose),
     shoulder midpoint (pose 11/12), each shoulder, and the opposite **upper arm** (shoulder–elbow midpoint, pose 13/14);
   - per hand: palm centre (mean of wrist, index MCP, pinky MCP) relative to nose, chin, shoulder midpoint
     and opposite upper arm; if hand landmarks are missing, fall back to the pose wrist (15/16) — hands
     near the body often drop out;
   - **hand-shape summary**: 5 fingertip-to-wrist distances / palm size (open vs fist/grip);
   - per-hand presence flag.
4. Interpolate missing frames, **resample each clip to a fixed 32 frames**.
5. **DTW comparator** (`src/dtw.py`): DTW distance (Sakoe-Chiba band) to **every library clip**, min over
   original and mirrored (left/right swapped) variant; **mean of 3 nearest per class**; softmax(−d/T).
   **NONE**: its distance = min(mean-3-nearest NONE clips, `d_none`), so NONE wins when every real class is
   too far. T and `d_none` calibrated on training data only.
6. Same decision logic. A final NONE is shown as "Not one of the 6 known signs — please try again".

### Service 2: Text/Drawing → Sign
- **Letters**: drawing canvas (multi-font/thickness template matcher, `src/handwriting.py`) or typing A/B/C.
- **Actions**: typing with **fuzzy matching** (`src/vocab.py`): alias table + `difflib`, e.g.
  "bp", "blood pressure", "pressure", "fever", "temperature", "temp", "bukhar", "stomach ache",
  "stomach pain", "tummy", "pet dard". Exact alias → 100%; fuzzy → similarity score through the same
  decision logic (so a typo can trigger "did you mean…"). Or **one button per action** → final.
- Output: text + reference image (letter) or **reference clip from `assets/isl/actions/`** (action) + optional TTS.

### Decision logic (shared, `src/decision.py`)
| Top confidence | Behaviour |
|---|---|
| `< 60%` (or out of range / NONE) | Ask the user to **retry** |
| `60% – <85%` | Show **top 2 candidates** to confirm (NONE never offered as a candidate) |
| `≥ 85%` | **Final answer**: text + ISL image/clip + optional offline TTS (`pyttsx3`, subprocess) |

Thresholds in `src/config.py` (`RETRY_BELOW = 0.60`, `FINAL_AT = 0.85`, `K_NEAREST = 3`).

## Held-out accuracy (must report)
- **Letters**: stratified split (`TEST_SIZE = 0.25`, `SEED = 42`); test vs train only; T from train split.
- **Actions**: **split by signer** — leave-one-signer-out: each fold tests one signer against the other
  signers' clips, with T/`d_none` calibrated (leave-one-signer-out) inside the training signers only.
  Report mean accuracy, per-class accuracy, confusion matrix incl. NONE, decision-band counts, NONE false
  accepts (NONE clip shown as an action) and false rejects.
- **Handwriting**: accuracy on `data/handwriting_test/`.
- The deployed app uses all clips/samples as the library; its calibration uses only that library (no test data).
- Caveat to state honestly: near-duplicate frames/clips from one session inflate random-split accuracy.

## File structure
```
app.py                      # Streamlit home + both services (+ sub-modes)
requirements.txt
src/
  config.py                 # classes, thresholds, paths, label-from-path
  vocab.py                  # 6-sign vocabulary, display text, aliases, fuzzy match, reference media
  extract.py                # letters: MediaPipe Hands + normalization; builds letter library
  comparator.py             # k=3 nearest-per-class comparator + train-only calibration (shared)
  handwriting.py            # canvas preprocessing + multi-font/thickness templates
  capture.py                # OpenCV countdown + 3 s clip with live Streamlit preview
  action_features.py        # Holistic -> body-relative per-frame features -> 32-frame resample
  extract_actions.py        # data/actions -> data/library/actions.npz
  dtw.py                    # DTW distance + action comparator (NONE handling)
  decision.py               # retry / confirm / final
  tts.py                    # offline TTS (pyttsx3 in a subprocess)
  train.py                  # letters + actions: split, calibrate, save models
  evaluate.py               # letters, actions (by signer), handwriting reports
data/raw/                   # letter images
data/actions/<CLASS>/       # <signer>_<n>.mp4
data/handwriting_test/      # real handwriting test set
data/library/               # generated
assets/isl/                 # A.png, B.png, C.png
assets/isl/actions/         # BLOOD_PRESSURE.mp4, FEVER.mp4, STOMACH_ACHE.mp4 (H.264)
models/  reports/           # generated
```

## Build order
**Letters (first — must work end to end before actions):**
1. Done: venv, requirements, letter modules, app pages smoke-tested on synthetic data.
2. Add `data/raw/`, run extract → check detection rate per letter.
3. train → evaluate → first real letter accuracy.
4. `assets/isl/{A,B,C}.png`; live webcam test of "Spell a letter"; handwriting test set.

**Actions (after letters work):**
5. `vocab.py` + Service 2 action typing/buttons + reference clips — cheapest visible win, no recognition needed.
6. `capture.py` in the app: countdown, live preview, 3 s record; verify webcam hand-off with `camera_input`.
7. `action_features.py` + `extract_actions.py`; check Holistic detection rates on `data/actions`.
8. `dtw.py` + calibration + leave-one-signer-out evaluation → action accuracy.
9. Service 1 "Show an action" sub-mode wired to the DTW comparator + decision logic.
10. Polish, README (scope + non-diagnostic framing), final accuracy report.
**Cut line if time runs out:** keep step 5 (text → action clip) and drop live action recognition.
