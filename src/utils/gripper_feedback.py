def feedback_current(
    effort: float,
    prev: float,
    gain: float,
    deadband: float,
    limit: int,
    alpha: float,
) -> int:
    target = max(0.0, abs(effort) - deadband) * gain
    target = min(target, float(limit))
    filtered = (1 - alpha) * prev + alpha * target
    return round(filtered)
