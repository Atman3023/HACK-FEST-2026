SignBridge — Hospital Communication Assistant
An assistive communication prototype designed to help bridge basic communication between people who use Indian Sign Language (ISL) and hospital reception or information-desk staff.

Project overview
Communication at a hospital reception can be difficult when a deaf or hard-of-hearing visitor uses Indian Sign Language (ISL) and the staff member does not understand it. SignBridge aims to provide a simple digital communication interface with two modes:
1. Sign-to-text / speech: The visitor shows an ISL hand gesture to a camera. The system detects the hand, predicts a supported sign, and displays the recognised result as text. The result may also be read aloud.
2. Text / speech-to-sign: A staff member types a short message or uses speech input. The system converts the input to text and displays the matching ISL sign from a reviewed, predefined sign library.
The initial prototype will focus on a small, clearly stated vocabulary. The supplied starter scope mentions ISL alphabet gestures A, B and C. The system must not claim to translate all of ISL or unrestricted conversations.
Intended setting and use case
Primary use case: a hospital reception or patient-assistance counter.

The prototype can demonstrate simple, non-clinical interactions such as identifying a demo service option or spelling a short identifier using the supported letters. It is not intended to collect medical histories, interpret symptoms, provide diagnoses, or replace a qualified interpreter—especially during emergencies or consent discussions.
User workflow
Mode 1 — Camera: ISL gesture to text / speech
1. The user selects Camera Mode and grants camera permission.
2. The browser captures frames from the webcam.
3. Hand tracking detects hand landmarks.
4. A trained classifier predicts one of the supported classes (initially A, B or C).
5. The interface displays the predicted letter and its confidence score.
6. If the prediction is uncertain, the interface asks the user to try again rather than presenting an unreliable result.
7. The user can add a confirmed letter to the message area and optionally play it using text-to-speech.
Mode 2 — Type or speak: text to ISL
1. The staff member selects Text / Speech Mode.
2. They type a short message, or use speech recognition where the browser supports it.
3. The application checks which words or letters are available in its approved sign library.
4. It displays the corresponding ISL image or video clips in sequence.
5. If a word is not supported, the interface clearly says so instead of pretending to translate it.
Important: ISL is a complete language with its own grammar and expression. Showing alphabet signs one after another is fingerspelling, not full sentence-level ISL translation. Word and sentence support must use verified ISL content and appropriate linguistic review.
Core features
- Two-way interface: gesture-to-text and text-to-sign.
- Live camera preview and hand-landmark detection.
- Initial A/B/C gesture classification.
- Prediction confidence display.
- Uncertain-gesture handling and retry prompt.
- Letter/message builder with add, undo and clear controls.
- Optional text-to-speech for recognised text.
- Optional speech-to-text for staff input.
- Curated ISL sign library for the reverse direction.
- Held-out test evaluation and documented limitations.
- Responsive interface suitable for a desktop or tablet kiosk.
