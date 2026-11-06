"""Offline text-to-speech -> WAV bytes, played in the visitor's BROWSER (st.audio autoplay).

Works the same locally and on Streamlit Community Cloud (no audio is played on the server, no cloud API):
  1. espeak-ng / espeak command line (Linux; installed on the cloud via packages.txt)
  2. pyttsx3.save_to_file (Windows SAPI / macOS NSSpeech), run in a subprocess
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave

_PYTTSX3 = ("import sys, pyttsx3; e = pyttsx3.init(); e.setProperty('rate', 165); "
            "e.save_to_file(sys.argv[1], sys.argv[2]); e.runAndWait()")


def _has_pyttsx3():
    """pyttsx3 is optional (Windows-only in requirements.txt); never imported in the app process."""
    try:
        import importlib.util
        return importlib.util.find_spec("pyttsx3") is not None
    except Exception:
        return False


def synthesize(text: str, timeout: float = 20.0):
    """-> WAV bytes, or None if no offline engine is available."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak:
            cmd = [espeak, "-v", "en", "-s", "150", "-w", path, text]
        elif _has_pyttsx3():
            cmd = [sys.executable, "-c", _PYTTSX3, text, path]
        else:
            return None
        subprocess.run(cmd, timeout=timeout, check=True, capture_output=True)
        data = open(path, "rb").read()
        return data if len(data) > 100 else None
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def with_padding(wav: bytes, n: int) -> bytes:
    """Same audio + n silent samples. Different bytes give a new media URL, so the browser autoplays it
    again ("Speak again") instead of reusing the finished player."""
    if n <= 0:
        return wav
    with wave.open(io.BytesIO(wav)) as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(frames + b"\x00" * (n * params.sampwidth * params.nchannels))
    return out.getvalue()
