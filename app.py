"""SignBridge: hospital reception assistant for ISL letters A, B and C.

Run:  streamlit run app.py
"""
import hashlib
import json

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src import config, handwriting, tts
from src.comparator import Comparator
from src.decision import decide
from src.extract import HandDetector, draw_landmarks, features

st.set_page_config(page_title="SignBridge", page_icon="🤟", layout="centered")
SCOPE = "Recognises only the ISL letters **A, B and C**. This is not a general ISL translator."


# ---------- cached resources ----------
@st.cache_resource
def sign_comparator():
    if not config.SIGN_MODEL_PATH.exists():
        return None
    m = np.load(config.SIGN_MODEL_PATH)
    # Library = all samples; T / d_max were calibrated on the training split only.
    return Comparator(m["X"], m["y"], float(m["T"]), float(m["d_max"]), int(m["k"]))


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


def isl_image(letter):
    for ext in (".png", ".jpg", ".jpeg"):
        p = config.ISL_ASSETS_DIR / f"{letter}{ext}"
        if p.exists():
            return str(p)
    return None


# ---------- shared UI ----------
def confidence_bars(confidences):
    st.markdown("**Confidence**")
    for letter in config.LETTERS:
        c = float(confidences.get(letter, 0.0))
        st.progress(min(max(c, 0.0), 1.0), text=f"{letter} — {100 * c:.0f}%")


def show_final(letter, key):
    st.success(f"## {letter}")
    img = isl_image(letter)
    if img:
        st.image(img, caption=f"ISL letter {letter}", width=260)
    else:
        st.caption(f"(Reference image missing: assets/isl/{letter}.png)")
    if st.button(f"🔊 Speak \"{letter}\"", key=f"speak_{key}"):
        if not tts.speak(f"Letter {letter}"):
            st.warning("Text-to-speech is unavailable on this machine.")


def show_decision(confidences, out_of_range, key):
    """Renders retry / confirm-top-2 / final. User confirmations are kept in session_state."""
    confidence_bars(confidences)
    confirmed = st.session_state.get(f"confirmed_{key}")
    if confirmed:
        st.info("Confirmed by user.")
        show_final(confirmed, key)
        return
    d = decide(confidences, out_of_range)
    if d.band == "retry":
        st.warning(f"🔁 {d.message}")
    elif d.band == "confirm":
        st.info(d.message)
        cols = st.columns(2)
        for col, (letter, conf) in zip(cols, d.ranked[:2]):
            if col.button(f"{letter}  ({100 * conf:.0f}%)", key=f"pick_{key}_{letter}",
                          use_container_width=True):
                st.session_state[f"confirmed_{key}"] = letter
                st.rerun()
        st.caption("Neither? Please try again.")
    else:
        show_final(d.top, key)


def go(page):
    st.session_state.page = page
    st.rerun()


# ---------- pages ----------
def home():
    st.title("🤟 SignBridge")
    st.write("Reception communication assistant for Indian Sign Language letters.")
    st.caption(SCOPE)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sign → Text / Speech")
        st.write("Show a hand sign to the camera.")
        if st.button("Start", key="go_sign", use_container_width=True):
            go("sign")
    with c2:
        st.subheader("Text / Drawing → Sign")
        st.write("Draw or type a letter to see its ISL sign.")
        if st.button("Start", key="go_text", use_container_width=True):
            go("text")

    r = report()
    with st.expander("Held-out accuracy"):
        if r and r.get("signs"):
            s = r["signs"]
            st.write(f"Sign recogniser: **{100 * s['top1_accuracy']:.1f}%** top-1 on "
                     f"{s['n']} held-out images.")
        else:
            st.write("Not evaluated yet — run `python -m src.evaluate`.")
        if r and r.get("handwriting"):
            h = r["handwriting"]
            st.write(f"Handwriting recogniser: **{100 * h['top1_accuracy']:.1f}%** top-1 on "
                     f"{h['n']} test drawings.")


def sign_page():
    st.header("Sign → Text / Speech")
    st.caption(SCOPE)
    comp = sign_comparator()
    if comp is None:
        st.error("Sign library not built. Run `python -m src.extract` then `python -m src.train`.")
        return
    shot = st.camera_input("Hold the sign in front of the camera, then take a photo")
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
             caption=f"{len(hands)} hand(s) detected", use_container_width=True)
    result = comp.predict(np.stack(features(hands)))
    show_decision(result["confidences"], result["out_of_range"], key)


def text_page():
    st.header("Text / Drawing → Sign")
    st.caption(SCOPE)
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
                show_decision(result["confidences"], result["out_of_range"], key)
    with type_tab:
        txt = st.text_input("Type a letter", max_chars=1, placeholder="A, B or C")
        if txt:
            letter = txt.strip().upper()
            if letter in config.LETTERS:
                confs = {l: 1.0 if l == letter else 0.0 for l in config.LETTERS}
                show_decision(confs, False, f"typed_{letter}")
            else:
                st.error("Only the letters A, B and C are supported.")


# ---------- router ----------
page = st.session_state.get("page", "home")
if page != "home":
    if st.sidebar.button("← Home"):
        go("home")
{"home": home, "sign": sign_page, "text": text_page}[page]()
