"""SignBridge: hospital reception assistant for 6 ISL signs (letters A, B, C; 3 action signs).

Run:  streamlit run app.py
"""
import hashlib
import json

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src import config, handwriting, tts, vocab
from src.comparator import Comparator
from src.decision import decide
from src.extract import HandDetector, draw_landmarks, features

st.set_page_config(page_title="SignBridge", page_icon="🤟", layout="centered")
SCOPE = ("Recognises only 6 ISL signs: letters **A, B, C** and the actions **blood pressure, fever, "
         "stomach ache**. Not a general ISL translator, and not a medical assessment.")
ACTION_MODEL_PATH = config.MODELS_DIR / "action_model.npz"


# ---------- cached resources ----------
@st.cache_resource
def sign_comparator():
    if not config.SIGN_MODEL_PATH.exists():
        return None
    m = np.load(config.SIGN_MODEL_PATH)
    # Library = all samples; T / d_max were calibrated on the training split only.
    return Comparator(m["X"], m["y"], float(m["T"]), float(m["d_max"]), int(m["k"]))


@st.cache_resource
def action_comparator():
    if not ACTION_MODEL_PATH.exists():
        return None
    from src.dtw import ActionComparator
    m = np.load(ACTION_MODEL_PATH)
    return ActionComparator(m["S"], m["y"], float(m["T"]), float(m["d_none"]), int(m["k"]))


@st.cache_resource
def hw_comparator():
    return handwriting.load_comparator()


@st.cache_resource
def detector():
    return HandDetector()


def report():
    try:
        return json.loads(config.REPORT_JSON.read_text())
    except Exception:
        return None


# ---------- shared UI ----------
def confidence_bars(confidences, title="Confidence"):
    st.markdown(f"**{title}**")
    for cls, c in confidences.items():
        c = float(c)
        st.progress(min(max(c, 0.0), 1.0), text=f"{vocab.DISPLAY.get(cls, cls)} — {100 * c:.0f}%")


def show_final(cls, key, speech):
    st.success(f"## {vocab.DISPLAY.get(cls, cls)}")
    media = vocab.reference_media(cls)
    if media is None:
        where = f"assets/isl/{cls}.png" if cls in config.LETTERS else f"assets/isl/actions/{cls}.mp4"
        st.caption(f"(Reference missing: {where})")
    elif cls in config.LETTERS:
        st.image(str(media), caption=f"ISL letter {cls}", width=260)
    else:
        st.video(str(media), loop=True, autoplay=True, muted=True)
        st.caption(f"ISL sign: {vocab.DISPLAY[cls]}")
    if st.button("🔊 Speak", key=f"speak_{key}"):
        if not tts.speak(speech):
            st.warning("Text-to-speech is unavailable on this machine.")


def speech_for(cls, visitor):
    name = vocab.DISPLAY[cls]
    if cls in config.LETTERS:
        return name
    return f"Visitor indicates: {name}" if visitor else name


def show_decision(confidences, out_of_range, key, visitor=True, title="Confidence"):
    """Renders retry / confirm-top-2 / final. User confirmations are kept in session_state."""
    confidence_bars(confidences, title)
    confirmed = st.session_state.get(f"confirmed_{key}")
    if confirmed:
        st.info("Confirmed by user.")
        show_final(confirmed, key, speech_for(confirmed, visitor))
        return
    d = decide(confidences, out_of_range)
    if d.band == "retry":
        st.warning(f"🔁 {d.message}")
    elif d.band == "confirm":
        st.info(d.message)
        cols = st.columns(2)
        for col, (cls, conf) in zip(cols, d.ranked[:2]):
            if col.button(f"{vocab.DISPLAY[cls]}  ({100 * conf:.0f}%)", key=f"pick_{key}_{cls}",
                          use_container_width=True):
                st.session_state[f"confirmed_{key}"] = cls
                st.rerun()
        st.caption("Neither? Please try again.")
    else:
        show_final(d.top, key, speech_for(d.top, visitor))


def go(page):
    st.session_state.page = page
    st.rerun()


# ---------- pages ----------
def home():
    st.title("🤟 SignBridge")
    st.write("Reception communication assistant for a small set of Indian Sign Language signs.")
    st.caption(SCOPE)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sign → Text / Speech")
        st.write("Visitor spells a letter or shows an action sign.")
        if st.button("Start", key="go_sign", use_container_width=True):
            go("sign")
    with c2:
        st.subheader("Text / Drawing → Sign")
        st.write("Staff draw or type to see the ISL sign.")
        if st.button("Start", key="go_text", use_container_width=True):
            go("text")

    r = report()
    with st.expander("Held-out accuracy"):
        shown = False
        if r and r.get("signs"):
            s = r["signs"]
            st.write(f"Letters: **{100 * s['top1_accuracy']:.1f}%** top-1 on {s['n']} held-out images.")
            shown = True
        if r and r.get("actions") and "top1_accuracy" in r["actions"]:
            a = r["actions"]
            st.write(f"Actions: **{100 * a['top1_accuracy']:.1f}%** top-1 on {a['n']} clips "
                     f"(each signer held out in turn).")
            shown = True
        if r and r.get("handwriting"):
            h = r["handwriting"]
            st.write(f"Handwriting: **{100 * h['top1_accuracy']:.1f}%** top-1 on {h['n']} test drawings.")
            shown = True
        if not shown:
            st.write("Not evaluated yet — run `python -m src.evaluate`.")


def letter_mode():
    comp = sign_comparator()
    if comp is None:
        st.error("Letter library not built. Run `python -m src.extract` then `python -m src.train`.")
        return
    shot = st.camera_input("Hold the letter sign in front of the camera, then take a photo")
    if shot is None:
        st.caption("Tips: whole hand(s) in frame, good light, plain background.")
        return
    data = shot.getvalue()
    key = hashlib.md5(data).hexdigest()[:10]
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    hands, res = detector().detect(bgr)
    if not hands:
        st.warning("🔁 No hand detected. Please retake the photo with your hand(s) fully visible.")
        return
    st.image(cv2.cvtColor(draw_landmarks(bgr, res), cv2.COLOR_BGR2RGB),
             caption=f"{len(hands)} hand(s) detected", use_column_width=True)
    result = comp.predict(np.stack(features(hands)))
    show_decision(result["confidences"], result["out_of_range"], key)


def action_mode():
    from src.action_features import mirror, process_frames, to_gif
    from src.capture import record_clip
    st.write("Sit so your **head, shoulders and upper body** are visible. After a 3-2-1 countdown, "
             f"perform the sign for {config.CLIP_SECONDS:.0f} seconds.")
    comp = action_comparator()
    if comp is None:
        st.info("Action library not built yet (record clips, then `python -m src.extract_actions` and "
                "`python -m src.train`). You can still record to check the landmark tracking.")
    live = st.empty()
    if st.button("🎬 Record action", type="primary", use_container_width=True):
        try:
            frames, fps = record_clip(live)
        except RuntimeError as e:
            live.empty()
            st.error(str(e))
            return
        live.info("Analysing the clip...")
        seq, stats, drawn = process_frames(frames, annotate=True)
        live.empty()
        result = None
        if seq is not None and comp is not None:
            result = comp.predict(seq, mirror(seq))
        st.session_state.action_clip = {
            "key": hashlib.md5(frames[len(frames) // 2].tobytes()).hexdigest()[:10],
            "gif": to_gif(drawn, fps, width=320, step=3), "stats": stats, "usable": seq is not None, "result": result}

    clip = st.session_state.get("action_clip")
    if not clip:
        return
    s = clip["stats"]
    st.image(clip["gif"], caption=f"Recorded clip with landmarks — body in {s['pose_rate']:.0%}, "
                                  f"hands in {s['hand_rate']:.0%} of frames", use_column_width=True)
    if not clip["usable"]:
        st.warning("🔁 Your upper body wasn't visible enough. Move back so your head and shoulders "
                   "are in frame, then record again.")
    elif clip["result"] is not None:
        r = clip["result"]
        show_decision(r["confidences"], r["out_of_range"], "act_" + clip["key"])


def sign_page():
    st.header("Sign → Text / Speech")
    st.caption(SCOPE)
    # Only one mode renders per run, so the browser camera and OpenCV never hold the webcam together.
    mode = st.radio("Mode", ["✋ Spell a letter", "🎬 Show an action"], horizontal=True,
                    label_visibility="collapsed")
    if mode.endswith("letter"):
        letter_mode()
    else:
        action_mode()


def letters_tab():
    draw_tab, type_tab = st.tabs(["✏️ Draw", "⌨️ Type"])
    with draw_tab:
        st.write("Draw **one** capital letter (A, B or C).")
        canvas = st_canvas(stroke_width=14, stroke_color="#000000", background_color="#FFFFFF",
                           height=280, width=280, drawing_mode="freedraw",
                           key=f"canvas_{st.session_state.get('canvas_n', 0)}")
        if st.button("Clear drawing"):
            st.session_state.canvas_n = st.session_state.get("canvas_n", 0) + 1
            st.rerun()
        if canvas.image_data is not None:
            v = handwriting.preprocess(handwriting.ink_from_rgba(canvas.image_data))
            if v is None:
                st.caption("Draw a letter above.")
            else:
                key = "draw_" + hashlib.md5(v.tobytes()).hexdigest()[:10]
                result = hw_comparator().predict(v)
                show_decision(result["confidences"], result["out_of_range"], key, visitor=False)
    with type_tab:
        txt = st.text_input("Type a letter", max_chars=1, placeholder="A, B or C")
        if txt:
            letter = txt.strip().upper()
            if letter in config.LETTERS:
                confs = {l: 1.0 if l == letter else 0.0 for l in config.LETTERS}
                show_decision(confs, False, f"typed_{letter}", visitor=False)
            else:
                st.error("Only the letters A, B and C are supported.")


def actions_tab():
    st.write("Tap an action, or type it (e.g. *bp*, *temperature*, *stomach pain*).")
    cols = st.columns(len(config.ACTIONS))
    for col, a in zip(cols, config.ACTIONS):
        if col.button(vocab.DISPLAY[a], key=f"btn_{a}", use_container_width=True):
            st.session_state.action_pick = a
            st.session_state.action_text = ""
    txt = st.text_input("Type an action", key="action_text", placeholder="e.g. blood pressure")
    if txt.strip():
        st.session_state.action_pick = None
        scores = vocab.match_action(txt)
        show_decision(scores, False, "typed_act_" + vocab.normalize(txt), visitor=False, title="Match")
    elif st.session_state.get("action_pick"):
        a = st.session_state.action_pick
        show_final(a, f"btn_{a}", vocab.DISPLAY[a])


def text_page():
    st.header("Text / Drawing → Sign")
    st.caption(SCOPE)
    letters, actions = st.tabs(["🔤 Letters", "🏥 Actions"])
    with letters:
        letters_tab()
    with actions:
        actions_tab()


# ---------- router ----------
page = st.session_state.get("page", "home")
if page != "home":
    if st.sidebar.button("← Home"):
        go("home")
{"home": home, "sign": sign_page, "text": text_page}[page]()
