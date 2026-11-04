"""Decision logic shared by both services: retry / confirm top-2 / final."""
from dataclasses import dataclass

from src import config


@dataclass
class Decision:
    band: str              # "retry" | "confirm" | "final"
    ranked: list           # [(letter, confidence), ...] best first
    message: str

    @property
    def top(self):
        return self.ranked[0][0]


def decide(confidences: dict, out_of_range: bool = False) -> Decision:
    ranked = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
    top_conf = ranked[0][1]
    if out_of_range:
        return Decision("retry", ranked, "This doesn't look like ISL A, B or C. Please try again.")
    if top_conf < config.RETRY_BELOW:
        return Decision("retry", ranked, "Low confidence. Please try again.")
    if top_conf < config.FINAL_AT:
        return Decision("confirm", ranked, "Did you mean one of these?")
    return Decision("final", ranked, f"Recognised: {ranked[0][0]}")
