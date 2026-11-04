"""Offline text-to-speech via pyttsx3 (Windows SAPI).

Runs in a separate process: pyttsx3 inside Streamlit's worker threads can hang on repeat calls.
"""
import subprocess
import sys

_SCRIPT = "import sys, pyttsx3; e = pyttsx3.init(); e.say(sys.argv[1]); e.runAndWait()"


def speak(text: str) -> bool:
    try:
        subprocess.Popen([sys.executable, "-c", _SCRIPT, text])
        return True
    except Exception:
        return False
