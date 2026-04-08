"""
Improvement index – tracks fitness trend over time using EWMA.
"""

EWMA_ALPHA = 0.1
IMPROVEMENT_LOOKBACK = 30  # activities


def compute_ewma(scores: list, alpha: float = EWMA_ALPHA) -> list:
    """
    Compute the exponentially-weighted moving average of scores.
    Expects scores ordered chronologically (oldest → newest).
    """
    if not scores:
        return []

    ewma = [scores[0]]
    for i in range(1, len(scores)):
        ewma.append(alpha * scores[i] + (1 - alpha) * ewma[i - 1])
    return ewma


def compute_improvement(activities: list) -> dict:
    """
    Compute the improvement index from a list of activities.

    Returns:
        {
            "current_ewma": float,
            "previous_ewma": float,
            "improvement_pct": float,
            "trend": "improving" | "flat" | "declining",
            "ewma_history": [ { "date", "name", "score", "ewma" }, ... ]
        }
    """
    if not activities:
        return {
            "current_ewma": 0,
            "previous_ewma": 0,
            "improvement_pct": 0,
            "trend": "flat",
            "ewma_history": [],
        }

    # Sort oldest → newest
    sorted_acts = sorted(activities, key=lambda a: a.get("start_date", ""))

    scores = [a.get("runny_score", 0) or 0 for a in sorted_acts]
    ewma = compute_ewma(scores)

    current_ewma = ewma[-1]
    prev_idx = max(0, len(ewma) - IMPROVEMENT_LOOKBACK - 1)
    previous_ewma = ewma[prev_idx]

    improvement_pct = (
        ((current_ewma - previous_ewma) / previous_ewma * 100)
        if previous_ewma > 0
        else 0
    )

    if improvement_pct > 2:
        trend = "improving"
    elif improvement_pct < -2:
        trend = "declining"
    else:
        trend = "flat"

    ewma_history = [
        {
            "date": a.get("start_date", ""),
            "name": a.get("name", ""),
            "score": a.get("runny_score", 0) or 0,
            "ewma": round(ewma[i], 1),
        }
        for i, a in enumerate(sorted_acts)
    ]

    return {
        "current_ewma": round(current_ewma, 1),
        "previous_ewma": round(previous_ewma, 1),
        "improvement_pct": round(improvement_pct, 1),
        "trend": trend,
        "ewma_history": ewma_history,
    }


def get_trend_color(trend: str) -> str:
    """Return a CSS color for the trend."""
    return {
        "improving": "#00e676",
        "declining": "#f44336",
        "flat": "#ffab00",
    }.get(trend, "#ffab00")


def get_trend_icon(trend: str) -> str:
    """Return an emoji icon for the trend."""
    return {
        "improving": "📈",
        "declining": "📉",
        "flat": "➡️",
    }.get(trend, "➡️")
