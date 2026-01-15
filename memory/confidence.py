from typing import Tuple


def update_confidence(current: float, stance: str, credibility_tier: str) -> Tuple[float, float]:
    delta = 0.0
    if stance == "support":
        delta += 0.05
        if credibility_tier.upper() == "A":
            delta += 0.10
    elif stance == "contradict":
        delta -= 0.05
        if credibility_tier.upper() == "A":
            delta -= 0.10
    new_conf = min(max(current + delta, 0.0), 1.0)
    return new_conf, delta
