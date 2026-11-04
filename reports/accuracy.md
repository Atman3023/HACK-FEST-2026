# SignBridge held-out accuracy

Scope: 6 ISL signs only (letters A, B, C; actions BLOOD_PRESSURE, FEVER, STOMACH_ACHE). Not a general ISL translator.

## Letters -> Text (held-out split, calibrated on train only)

_No letter model yet (needs data/raw)._

## Actions -> Text (held-out by signer)

_No action library yet. Record clips, then run `python -m src.extract_actions`._

## Handwriting -> Sign (data/handwriting_test)

_No test set yet (add images to data/handwriting_test/<A|B|C>/)._
