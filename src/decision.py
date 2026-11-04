"""Decision logic shared by both services: retry / confirm top-2 / final."""
from dataclasses import dataclass

from src import config

NONE = "NONE"


@dataclass
class Decision:
    band: str              # "retry" | "confirm" | "final"
    ranked: list           # [(class, confidence), ...] best first; NONE is never a confirm candidate
    message: str

    @property
    def top(self):
        return self.ranked[0][0]


def decide(confidences: dict, out_of_range: bool = False) -> Decision:
    ranked = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
    top, top_conf = ranked[0]
    if out_of_range:
        return Decision("retry", ranked, "This doesn't look like one of the known signs. Please try again.")
    if top == NONE:
        return Decision("retry", ranked, "Not one of the known signs. Please try again.")
    if top_conf < config.RETRY_BELOW:
        return Decision("retry", ranked, "Low confidence. Please try again.")
    if top_conf < config.FINAL_AT:
        # keep NONE out of the two buttons offered to the user
        signs = [r for r in ranked if r[0] != NONE] + [r for r in ranked if r[0] == NONE]
        return Decision("confirm", signs, "Did you mean one of these?")
    return Decision("final", ranked, f"Recognised: {top}")
