import math

from research_radar.config import RadarScoring


def radar_score(
    scores: list[float], coefficients: list[float], configuration: RadarScoring
) -> float:
    """Combine quality, not quantity, using a fixed-denominator peak/tail formula.

    Sort scores descending and transform each with
    q(s) = max(0, (s - quality_floor) / (100 - quality_floor)) ** quality_power.
    The first len(coefficients)-1 positions are individual weighted slots; the
    final coefficient weights the MEAN transformed quality of all remaining
    scores. Missing slots contribute nothing. Return
    100 * (peak_share * best/100 + (1-peak_share) * strength/sum(coefficients)).
    The pooled tail is bounded independently of its size.
    """
    if (
        len(coefficients) < 2
        or any(not math.isfinite(c) or c < 0 for c in coefficients)
        or coefficients[0] <= 0
        or any(a < b for a, b in zip(coefficients, coefficients[1:], strict=False))
        or not math.isfinite(sum(coefficients))
    ):
        raise ValueError("Coefficients must be finite, nonnegative, and nonincreasing")
    if any(not math.isfinite(s) or not 0 <= s <= 100 for s in scores):
        raise ValueError("Scores must be finite and in the range 0..100")
    floor, power, peak = (
        configuration.quality_floor,
        configuration.quality_power,
        configuration.peak_share,
    )
    if (
        not all(math.isfinite(x) for x in (floor, power, peak))
        or not 0 <= floor < 100
        or power < 1
        or not 0 <= peak <= 1
    ):
        raise ValueError("Invalid radar scoring configuration")
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)
    quality = [math.pow(max(0.0, (score - floor) / (100 - floor)), power) for score in ordered]
    slots = len(coefficients) - 1
    strength = sum(c * q for c, q in zip(coefficients[:slots], quality[:slots], strict=False))
    tail = quality[slots:]
    if tail:
        strength += coefficients[-1] * (sum(tail) / len(tail))
    return 100 * (peak * (ordered[0] / 100) + (1 - peak) * strength / sum(coefficients))
