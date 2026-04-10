"""
Improvement index v2 – tracks fitness trend over time using
window-based metric comparisons instead of EWMA of scores.

Computes an Improvement Index (0–100) by comparing key metrics
between two consecutive time windows (recent vs. previous).
"""
from datetime import datetime, timedelta, timezone
import math

# ─── Configuration ────────────────────────────────────────

DEFAULT_TREND_WINDOW_DAYS = 28
EWMA_ALPHA = 0.1  # Kept for chart compatibility
MIN_ACTIVITIES_PER_WINDOW = 3


# ─── Helpers ──────────────────────────────────────────────

def _parse_date(date_str):
    """Parse an ISO date string to a naive datetime."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _split_windows(activities, window_days=DEFAULT_TREND_WINDOW_DAYS):
    """
    Split activities into two windows:
    - recent:   last `window_days` days
    - previous: `window_days+1` to `2*window_days` days ago

    Activities must be passed in any order (sorted internally).
    Returns (recent_list, previous_list).
    """
    # Use the most recent activity date as 'now' so data isn't lost if the user hasn't synced recently
    latest_dt = None
    for a in activities:
        dt = _parse_date(a.get("start_date", ""))
        if dt and (latest_dt is None or dt > latest_dt):
            latest_dt = dt
            
    now = latest_dt if latest_dt else datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff_recent = now - timedelta(days=window_days)
    cutoff_previous = now - timedelta(days=window_days * 2)

    recent = []
    previous = []

    for a in activities:
        dt = _parse_date(a.get("start_date", ""))
        if not dt:
            continue
        if dt >= cutoff_recent:
            recent.append(a)
        elif dt >= cutoff_previous:
            previous.append(a)

    return recent, previous


def _median(values):
    """Compute the median of a list of numbers."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _sigmoid_scale(value, center=0.0, spread=1.0):
    """
    Map a raw value to 0–100 using a logistic (sigmoid) curve.
    - center: the value that maps to 50
    - spread: controls the steepness (higher = more gradual)
    Returns a float in [0, 100].
    """
    try:
        x = (value - center) / spread
        return 100.0 / (1.0 + math.exp(-x))
    except (OverflowError, ZeroDivisionError):
        return 50.0


def _metric_trend(recent_activities, previous_activities, key, invert=False):
    """
    Compute the trend (improvement delta) for a single metric.

    - invert=True: lower value = better (e.g. pace in sec/km)
    - invert=False: higher value = better (e.g. efficiency)

    Returns the raw delta (positive = improving) or None if not enough data.
    """
    recent_vals = [a.get(key) for a in recent_activities if a.get(key) is not None and a.get(key) > 0]
    prev_vals = [a.get(key) for a in previous_activities if a.get(key) is not None and a.get(key) > 0]

    if len(recent_vals) < 2 or len(prev_vals) < 2:
        return None

    recent_med = _median(recent_vals)
    prev_med = _median(prev_vals)

    if invert:
        # Lower is better → positive delta when recent < previous
        return prev_med - recent_med
    else:
        # Higher is better → positive delta when recent > previous
        return recent_med - prev_med


def _compute_efficiency_values(activities):
    """Compute cardiac efficiency for a list of activities."""
    efficiencies = []
    for a in activities:
        pace = a.get("pace", 0)
        hr = a.get("average_heartrate")
        if pace > 0 and hr and hr > 0:
            speed = 1000.0 / pace
            eff = (speed / hr) * 1000
            efficiencies.append(eff)
    return efficiencies


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


# ─── Public API ───────────────────────────────────────────

def compute_improvement(activities: list, window_days: int = DEFAULT_TREND_WINDOW_DAYS) -> dict:
    """
    Compute the improvement index from a list of activities.

    Returns:
        {
            "improvement_index": float (0–100),
            "trend": "improving" | "flat" | "declining",
            "window_days": int,
            "pace_trend": float | None,
            "efficiency_trend": float | None,
            "volume_trend": float | None,
            "breakdown": { ... },
            "ewma_history": [ { "date", "name", "score", "ewma" }, ... ]
        }
    """
    empty = {
        "improvement_index": 50,
        "trend": "flat",
        "window_days": window_days,
        "pace_trend": None,
        "efficiency_trend": None,
        "volume_trend": None,
        "breakdown": {},
        "ewma_history": [],
        # Legacy fields for backward compat
        "current_ewma": 0,
        "previous_ewma": 0,
        "improvement_pct": 0,
    }

    if not activities:
        return empty

    # ── EWMA history (for chart backward compatibility) ──
    sorted_acts = sorted(activities, key=lambda a: a.get("start_date", ""))
    scores = [a.get("runny_score", 0) or 0 for a in sorted_acts]
    ewma = compute_ewma(scores)

    ewma_history = [
        {
            "date": a.get("start_date", ""),
            "name": a.get("name", ""),
            "score": a.get("runny_score", 0) or 0,
            "ewma": round(ewma[i], 1),
            "pace": a.get("pace", 0),
            "moving_time": a.get("moving_time", 0),
            "distance": a.get("distance", 0),
            "average_heartrate": a.get("average_heartrate"),
            "max_heartrate": a.get("max_heartrate"),
        }
        for i, a in enumerate(sorted_acts)
    ]

    # Legacy EWMA values
    current_ewma = ewma[-1] if ewma else 0
    # Use window_days instead of a hardcoded 30 to respect user's time window selection
    prev_idx = max(0, len(ewma) - window_days - 1)
    previous_ewma = ewma[prev_idx] if ewma else 0
    improvement_pct = (
        ((current_ewma - previous_ewma) / previous_ewma * 100)
        if previous_ewma > 0
        else 0
    )

    # ── Window-based trend analysis ──
    recent, previous = _split_windows(activities, window_days)

    if len(recent) < MIN_ACTIVITIES_PER_WINDOW or len(previous) < MIN_ACTIVITIES_PER_WINDOW:
        # Not enough data for trend analysis → fall back to EWMA-based judgment
        if improvement_pct > 2:
            trend = "improving"
        elif improvement_pct < -2:
            trend = "declining"
        else:
            trend = "flat"

        return {
            "improvement_index": 50 + min(25, max(-25, improvement_pct)),
            "trend": trend,
            "window_days": window_days,
            "pace_trend": None,
            "efficiency_trend": None,
            "volume_trend": None,
            "breakdown": {"note": f"Need ≥{MIN_ACTIVITIES_PER_WINDOW} activities in each window"},
            "ewma_history": ewma_history,
            "current_ewma": round(current_ewma, 1),
            "previous_ewma": round(previous_ewma, 1),
            "improvement_pct": round(improvement_pct, 1),
        }

    breakdown = {}

    # 1. Pace trend (lower pace = faster = improving)
    pace_delta = _metric_trend(recent, previous, "pace", invert=True)
    pace_score = 50.0
    if pace_delta is not None:
        # Typical meaningful change: ±15 sec/km
        pace_score = _sigmoid_scale(pace_delta, center=0, spread=8)
        breakdown["pace"] = {
            "delta_sec_per_km": round(pace_delta, 1),
            "score": round(pace_score, 1),
        }

    # 2. Cardiac efficiency trend (higher = better)
    recent_effs = _compute_efficiency_values(recent)
    prev_effs = _compute_efficiency_values(previous)
    eff_delta = None
    eff_score = 50.0
    if len(recent_effs) >= 2 and len(prev_effs) >= 2:
        eff_delta = _median(recent_effs) - _median(prev_effs)
        # Typical meaningful change: ±1.0 efficiency units
        eff_score = _sigmoid_scale(eff_delta, center=0, spread=0.5)
        breakdown["efficiency"] = {
            "delta": round(eff_delta, 2),
            "score": round(eff_score, 1),
        }

    # 3. Volume trend (total distance ratio)
    recent_dist = sum(a.get("distance", 0) for a in recent)
    prev_dist = sum(a.get("distance", 0) for a in previous)
    vol_ratio = None
    vol_score = 50.0
    if prev_dist > 0:
        vol_ratio = recent_dist / prev_dist
        # Map ratio: 0.5 → low, 1.0 → neutral, 1.5 → high
        vol_score = _sigmoid_scale(vol_ratio, center=1.0, spread=0.25)
        breakdown["volume"] = {
            "recent_km": round(recent_dist / 1000, 1),
            "previous_km": round(prev_dist / 1000, 1),
            "ratio": round(vol_ratio, 2),
            "score": round(vol_score, 1),
        }

    # ── Weighted blend ──
    weights = {"pace": 0.40, "efficiency": 0.40, "volume": 0.20}
    total_weight = 0
    weighted_sum = 0

    if pace_delta is not None:
        weighted_sum += pace_score * weights["pace"]
        total_weight += weights["pace"]
    if eff_delta is not None:
        weighted_sum += eff_score * weights["efficiency"]
        total_weight += weights["efficiency"]
    if vol_ratio is not None:
        weighted_sum += vol_score * weights["volume"]
        total_weight += weights["volume"]

    if total_weight > 0:
        improvement_index = weighted_sum / total_weight
    else:
        improvement_index = 50.0

    improvement_index = round(min(100, max(0, improvement_index)), 1)

    if improvement_index > 55:
        trend = "improving"
    elif improvement_index < 45:
        trend = "declining"
    else:
        trend = "flat"

    return {
        "improvement_index": improvement_index,
        "trend": trend,
        "window_days": window_days,
        "pace_trend": round(pace_delta, 1) if pace_delta is not None else None,
        "efficiency_trend": round(eff_delta, 2) if eff_delta is not None else None,
        "volume_trend": round(vol_ratio, 2) if vol_ratio is not None else None,
        "breakdown": breakdown,
        "ewma_history": ewma_history,
        # Legacy fields
        "current_ewma": round(current_ewma, 1),
        "previous_ewma": round(previous_ewma, 1),
        "improvement_pct": round(improvement_pct, 1),
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
