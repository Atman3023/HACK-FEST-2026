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
    if "fit_x" not in m.files:   # model from an older build
        return None
    params = {"T": float(m["T"]), "fit_x": m["fit_x"], "fit_y": m["fit_y"]}
    comp = ActionComparator(m["S"], m["y"], m["parent"], params, int(m["k"]), float(m["min_motion"]))
    comp.meta = {"files": m["files"], "signer": m["signer"], "source": m["source"], "aug": m["aug"]}
    return comp


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


def placeholder_card(cls):
    st.markdown(
        f"""<div style="border:2px dashed #9aa5b1;border-radius:12px;padding:28px 16px;text-align:center;
        color:#52606d;max-width:360px;margin:4px 0 8px 0">
        <div style="font-size:2.2rem">🎬</div>
        <div style="font-weight:600;margin-top:6px">{vocab.DISPLAY[cls]}</div>
        <div style="margin-top:4px">Reference clip to be recorded</div></div>""",
        unsafe_allow_html=True)


def show_final(cls, key, speech):
    st.success(f"## {vocab.DISPLAY.get(cls, cls)}")
    media = vocab.reference_media(cls)
    if cls in config.LETTERS:
        if media is None:
            st.caption(f"(Reference image missing: assets/isl/{cls}.jpg)")
        else:
            st.image(str(media), caption=f"ISL letter {cls}", width=300)
    elif media is None:
        placeholder_card(cls)
    else:
        st.video(str(media), loop=True, autoplay=True, muted=True)
        st.caption(f"ISL sign: {vocab.DISPLAY[cls]} (reference clip)")
    # speak automatically once per result; reruns of the same result stay quiet
    spoken = st.session_state.setdefault("spoken", set())
    if key not in spoken:
        spoken.add(key)
        if not tts.speak(speech):
            st.warning("Text-to-speech is unavailable on this machine.")
    if st.button("🔊 Speak again", key=f"speak_{key}"):
        tts.speak(speech)


def speech_for(cls, visitor):
    name = vocab.DISPLAY[cls]
    if cls in config.LETTERS:
        return name
    return f"Visitor indicates: {name}" if visitor else name


def show_decision(confidences, out_of_range, key, visitor=True, title="Confidence", reason=None):
    """Renders retry / confirm-top-2 / final. User confirmations are kept in session_state."""
    if out_of_range:
        title = "Closest match (rejected — not used)"
    confidence_bars(confidences, title)
    confirmed = st.session_state.get(f"confirmed_{key}")
    if confirmed:
        st.info("Confirmed by user.")
        show_final(confirmed, key, speech_for(confirmed, visitor))
        return
    d = decide(confidences, out_of_range)
    if d.band == "retry" and (out_of_range or d.top == "NONE"):
        st.error("🚫 **Not a known sign** — please try again." + (f" ({reason})" if reason else ""))
    elif d.band == "retry":
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
            if "error" in s:
                st.write("Letters: held-out accuracy can't be measured yet (too few photos per letter).")
            else:
                st.write(f"Letters: **{100 * s['top1_accuracy']:.1f}%** top-1 on {s['n']} held-out images.")
            shown = True
        if r and r.get("actions") and r["actions"].get("top1_accuracy") is not None:
            a = r["actions"]
            st.write(f"Actions: **{100 * a['top1_accuracy']:.1f}%** top-1 on {a['n']} clips "
                     f"({', '.join(a['measurable'])}; each signer held out in turn).")
            for c in a.get("unmeasurable", []):
                st.write(f"{vocab.DISPLAY[c]}: can't be measured yet — only one source video.")
            shown = True
        if r and r.get("handwriting"):
            h = r["handwriting"]
            st.write(f"Handwriting: **{100 * h['top1_accuracy']:.1f}%** top-1 on {h['n']} test drawings.")
            shown = True
        if not shown:
            st.write("Not evaluated yet — run `python scripts/build_all.py`.")


def letter_mode():
    comp = sign_comparator()
    if comp is None:
        st.error("Letter library not built. Run `python scripts/build_all.py`.")
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


def speak_once(key, speech):
    """Speak automatically once per result; reruns of the same result stay quiet."""
    spoken = st.session_state.setdefault("spoken", set())
    if key not in spoken:
        spoken.add(key)
        if not tts.speak(speech):
            st.warning("Text-to-speech is unavailable on this machine.")


def big_text(text, color="#1b5e20", sub=None):
    st.markdown(f"""<div style="font-size:3.2rem;font-weight:800;line-height:1.15;color:{color};
                margin:12px 0 4px 0">{text}</div>""", unsafe_allow_html=True)
    if sub:
        st.caption(sub)


def action_result(r, key):
    """Service 1 output: only the result, as large text. Final answers are spoken automatically."""
    confirmed = st.session_state.get(f"confirmed_{key}")
    d = decide(r["confidences"], r["out_of_range"])
    final = confirmed or (d.top if d.band == "final" else None)
    if final:
        speech = speech_for(final, True)
        big_text(vocab.DISPLAY[final], sub="Visitor indicates this sign" + (" (confirmed)" if confirmed else ""))
        speak_once(key + final, speech)
        if st.button("🔊 Speak again", key=f"speak_{key}"):
            tts.speak(speech)
    elif d.band == "retry" and (r["out_of_range"] or d.top == "NONE"):
        big_text("Not a known sign", "#b71c1c", sub="Please try again." +
                 (f" Reason: {r['reject_reason']}." if r.get("reject_reason") else ""))
    elif d.band == "retry":
        big_text("Please try again", "#e65100", sub="Low confidence.")
    else:
        big_text("Did you mean…?", "#0d47a1")
        cols = st.columns(2)
        for col, (cls, conf) in zip(cols, d.ranked[:2]):
            if col.button(vocab.DISPLAY[cls], key=f"pick_{key}_{cls}", use_container_width=True):
                st.session_state[f"confirmed_{key}"] = cls
                st.rerun()


def action_debug(clip, comp):
    r, s = clip["result"], clip["stats"]
    with st.expander("Debug", expanded=False):
        st.image(clip["gif"], caption=f"Recorded clip with landmarks — body in {s['pose_rate']:.0%}, "
                                      f"hands in {s['hand_rate']:.0%} of frames", width=320)
        if r is None:
            return
        confidence_bars(r["confidences"], "Confidence (ranking x distance)")
        st.caption("Class distance (nearest segment): " + ", ".join(
            f"{vocab.DISPLAY[c]} {v:.2f}" for c, v in r["distances"].items() if np.isfinite(v))
            + (f" · hand motion {r['motion']:.2f} (min {comp.min_motion:.2f})" if r.get("motion") is not None else ""))
        d = r["dist_all"]
        order = np.argsort(d)[:5]
        m = comp.meta
        st.markdown("**5 nearest library segments**")
        st.table([{"class": str(comp.y[i]), "signer": str(m["signer"][i]), "source video": str(m["source"][i]),
                   "clip": str(m["files"][i]) + (" (augmented)" if m["aug"][i] else ""),
                   "DTW distance": f"{d[i]:.3f}"} for i in order])


def collect_panel(clip):
    """Save the last recording into the library, rebuild the library, show clip counts."""
    from src import collect
    with st.expander("📼 Add this recording to the library", expanded=False):
        if clip and clip.get("frames"):
            signer = st.text_input("Signer name", key="collect_signer", placeholder="e.g. atman")
            saved = st.session_state.get(f"saved_{clip['key']}")
            if saved:
                st.success(f"Saved: {saved}")
            else:
                st.write("Save as:")
                cols = st.columns(len(config.ACTION_CLASSES))
                for col, cls in zip(cols, config.ACTION_CLASSES):
                    label = "None" if cls == "NONE" else vocab.DISPLAY[cls].replace(" check", "")
                    if col.button(label, key=f"save_{cls}_{clip['key']}", use_container_width=True):
                        try:
                            path = collect.save_clip(clip["frames"], clip["fps"], cls, signer)
                            st.session_state[f"saved_{clip['key']}"] = str(path.relative_to(config.ROOT))
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
        else:
            st.caption("Record an action first, then save it here with its correct label.")
        counts = collect.clip_counts()
        st.markdown("**Clips per class** (originals, augmented copies not counted)")
        st.table([{"class": vocab.DISPLAY.get(c, c) if c != "NONE" else "None (not a sign)",
                   "clips": sum(cn.values()), "signers": ", ".join(f"{k} ({v})" for k, v in sorted(cn.items()))}
                  for c, cn in counts.items()])
        if st.button("🔄 Rebuild library", key="rebuild"):
            with st.spinner("Extracting features of new clips and rebuilding the library..."):
                ok, log = collect.rebuild_library()
            action_comparator.clear()
            (st.success if ok else st.error)("Library rebuilt and reloaded." if ok else "Rebuild failed — see log.")
            st.code(log[-3000:] or "(no output)")


def action_mode():
    from src.action_features import mirror, process_frames, to_gif
    from src.capture import record_clip
    st.write("Sit so your **head, shoulders and upper body** are visible. After a 3-2-1 countdown, "
             f"perform the sign for {config.CLIP_SECONDS:.0f} seconds.")
    comp = action_comparator()
    if comp is None:
        st.info("Action library not built yet. Record and save clips below, then press **Rebuild library**. "
                "You can still record to check the landmark tracking.")
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
            "gif": to_gif(drawn, fps, width=320, step=3), "stats": stats, "usable": seq is not None,
            "result": result, "frames": frames, "fps": fps}

    clip = st.session_state.get("action_clip")
    if clip:
        if not clip["usable"]:
            big_text("Please try again", "#e65100",
                     sub="Your upper body wasn't visible enough. Move back so your head and shoulders are in frame.")
        elif clip["result"] is not None:
            action_result(clip["result"], "act_" + clip["key"])
        if comp is not None:
            action_debug(clip, comp)
    collect_panel(clip)


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


def type_tab():
    st.write("Type a sign, e.g. **bp**, *check blood pressure*, *temperature*, *bukhar*, *tummy pain*, "
             "*pet dard*, or a letter **a / b / c**. Typos are fine.")
    txt = st.text_input("Type here", key="typed_text", placeholder="e.g. blood pressure")
    if not txt.strip():
        return
    scores = vocab.match(txt)
    if not vocab.in_vocabulary(scores):
        st.error(f"**Not in vocabulary:** “{txt.strip()}”")
        st.write("Supported signs: " + ", ".join(f"**{s}**" for s in vocab.supported_list()))
        confidence_bars(scores, "Match")
        return
    show_decision(scores, False, "typed_" + vocab.normalize(txt), visitor=False, title="Match")


def draw_tab():
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


def buttons_tab():
    st.write("Tap an action to show its ISL sign.")
    cols = st.columns(len(config.ACTIONS))
    for col, a in zip(cols, config.ACTIONS):
        if col.button(vocab.DISPLAY[a], key=f"btn_{a}", use_container_width=True):
            st.session_state.action_pick = a
            st.session_state.pick_n = st.session_state.get("pick_n", 0) + 1
    a = st.session_state.get("action_pick")
    if a:
        show_final(a, f"btn_{a}_{st.session_state.get('pick_n', 0)}", vocab.DISPLAY[a])


def text_page():
    st.header("Text / Drawing → Sign")
    st.caption(SCOPE)
    t_type, t_draw, t_buttons = st.tabs(["⌨️ Type", "✏️ Draw a letter", "🏥 Pick an action"])
    with t_type:
        type_tab()
    with t_draw:
        draw_tab()
    with t_buttons:
        buttons_tab()


# ---------- router ----------
page = st.session_state.get("page", "home")
if page != "home":
    if st.sidebar.button("← Home"):
        go("home")
{"home": home, "sign": sign_page, "text": text_page}[page]()
