"""
Scoring engine – assigns each activity a score (0-100) relative to the
athlete's recent baseline using weighted percentile ranking.
"""
from datetime import datetime, timedelta

# ─── Configuration ────────────────────────────────────────

BASELINE_WINDOW_DAYS = 90
MIN_ACTIVITIES_FOR_PERCENTILE = 5

WEIGHTS = {
    "distance": 0.20,
    "duration": 0.10,
    "pace": 0.20,
    "elevation": 0.15,
    "avg_hr": 0.10,
    "max_hr": 0.05,
    "rpe": 0.10,
    "training_type": 0.10,
}

TYPE_MULTIPLIERS = {
    "easy": 0.80,
    "moderate": 1.00,
    "tempo": 1.10,
    "long": 1.15,
    "intervals": 1.20,
    "repetitions": 1.20,
    "race": 1.30,
}

# Approximate medians for a recreational runner (fallback when < 5 activities)
FALLBACK_MEDIANS = {
    "distance": 7000,    # 7 km in meters
    "duration": 2400,    # 40 min in seconds
    "pace": 360,         # 6:00 /km in seconds
    "elevation": 100,    # meters
    "avg_hr": 150,       # bpm
    "max_hr": 175,       # bpm
}


# ─── Helpers ──────────────────────────────────────────────

def _percentile_rank(value, sorted_values, invert=False):
    """
    Compute the percentile rank (0–100) of `value` within `sorted_values`.
    If invert=True, lower value = higher percentile (e.g. pace: faster is better).
    """
    if not sorted_values:
        return 50.0

    count = 0
    for v in sorted_values:
        if invert:
            if value <= v:
                count += 1
        else:
            if value >= v:
                count += 1

    pct = (count / len(sorted_values)) * 100
    return min(100.0, max(0.0, pct))


def _fallback_percentile(value, median, invert=False):
    """Estimate percentile using an absolute median benchmark."""
    if not value or not median:
        return 50.0
    ratio = (median / value) if invert else (value / median)
    return min(100.0, max(0.0, ratio * 50))


def _get_baseline(all_activities):
    """Filter activities to the last BASELINE_WINDOW_DAYS."""
    cutoff = datetime.utcnow() - timedelta(days=BASELINE_WINDOW_DAYS)
    baseline = []
    for a in all_activities:
        try:
            dt = datetime.fromisoformat(a["start_date"].replace("Z", "+00:00"))
            if dt.replace(tzinfo=None) >= cutoff:
                baseline.append(a)
        except (ValueError, KeyError):
            continue
    return baseline


# ─── Public API ───────────────────────────────────────────

def score_activity(activity: dict, all_activities: list) -> tuple:
    """
    Score a single activity relative to the baseline.
    Returns (total_score: int, breakdown: dict).
    """
    baseline = _get_baseline(all_activities)
    use_percentile = len(baseline) >= MIN_ACTIVITIES_FOR_PERCENTILE

    # Build sorted arrays from baseline
    distances = sorted(a.get("distance", 0) for a in baseline)
    durations = sorted(a.get("moving_time", 0) for a in baseline)
    paces = sorted(a.get("pace", 0) for a in baseline if a.get("pace", 0) > 0)
    elevations = sorted(a.get("total_elevation_gain", 0) for a in baseline)
    avg_hrs = sorted(a["average_heartrate"] for a in baseline if a.get("average_heartrate"))
    max_hrs = sorted(a["max_heartrate"] for a in baseline if a.get("max_heartrate"))

    breakdown = {}
    weighted_sum = 0.0
    active_weight = 0.0

    # Distance (higher = harder workout)
    dist = activity.get("distance", 0)
    p = _percentile_rank(dist, distances) if use_percentile else _fallback_percentile(dist, FALLBACK_MEDIANS["distance"])
    breakdown["distance"] = round(p, 1)
    weighted_sum += p * WEIGHTS["distance"]
    active_weight += WEIGHTS["distance"]

    # Duration
    dur = activity.get("moving_time", 0)
    p = _percentile_rank(dur, durations) if use_percentile else _fallback_percentile(dur, FALLBACK_MEDIANS["duration"])
    breakdown["duration"] = round(p, 1)
    weighted_sum += p * WEIGHTS["duration"]
    active_weight += WEIGHTS["duration"]

    # Pace (lower = faster = better → invert)
    pace = activity.get("pace", 0)
    if pace > 0:
        p = _percentile_rank(pace, paces, invert=True) if use_percentile else _fallback_percentile(pace, FALLBACK_MEDIANS["pace"], invert=True)
        breakdown["pace"] = round(p, 1)
        weighted_sum += p * WEIGHTS["pace"]
        active_weight += WEIGHTS["pace"]

    # Elevation
    elev = activity.get("total_elevation_gain", 0)
    p = _percentile_rank(elev, elevations) if use_percentile else _fallback_percentile(elev, FALLBACK_MEDIANS["elevation"])
    breakdown["elevation"] = round(p, 1)
    weighted_sum += p * WEIGHTS["elevation"]
    active_weight += WEIGHTS["elevation"]

    # Avg HR
    avg_hr = activity.get("average_heartrate")
    if avg_hr:
        p = _percentile_rank(avg_hr, avg_hrs, invert=True) if use_percentile else _fallback_percentile(avg_hr, FALLBACK_MEDIANS["avg_hr"], invert=True)
        breakdown["avg_hr"] = round(p, 1)
        weighted_sum += p * WEIGHTS["avg_hr"]
        active_weight += WEIGHTS["avg_hr"]

    # Max HR
    max_hr = activity.get("max_heartrate")
    if max_hr:
        p = _percentile_rank(max_hr, max_hrs, invert=True) if use_percentile else _fallback_percentile(max_hr, FALLBACK_MEDIANS["max_hr"], invert=True)
        breakdown["max_hr"] = round(p, 1)
        weighted_sum += p * WEIGHTS["max_hr"]
        active_weight += WEIGHTS["max_hr"]

    # RPE (user-entered 1–10 → scale to 0–100)
    rpe = activity.get("rpe")
    if rpe and rpe > 0:
        rpe_pct = (rpe / 10) * 100
        breakdown["rpe"] = round(rpe_pct, 1)
        weighted_sum += rpe_pct * WEIGHTS["rpe"]
        active_weight += WEIGHTS["rpe"]

    # Normalize to active weights (weighted_sum is already 0–100 scale)
    raw_score = (weighted_sum / active_weight) if active_weight > 0 else 50.0

    # Training type multiplier
    training_type = (activity.get("training_type") or "moderate").lower()
    multiplier = TYPE_MULTIPLIERS.get(training_type, 1.0)
    breakdown["type_multiplier"] = multiplier

    final_score = int(min(100, max(0, round(raw_score * multiplier))))
    breakdown["raw_score"] = round(raw_score, 1)

    return final_score, breakdown


def score_all_activities(activities: list) -> list:
    """Score all activities in bulk, returning them with runny_score set."""
    result = []
    for a in activities:
        score, _ = score_activity(a, activities)
        a_copy = dict(a)
        a_copy["runny_score"] = score
        result.append(a_copy)
    return result


def get_score_color(score):
    """Return a CSS color for the given score."""
    if score is None:
        return "#6b6f88"
    if score >= 85:
        return "#00e676"
    if score >= 70:
        return "#ff9800"
    if score >= 50:
        return "#ffeb3b"
    if score >= 30:
        return "#ff5722"
    return "#f44336"


def get_score_label(score):
    """Return a human label for the given score."""
    if score is None:
        return "—"
    if score >= 85:
        return "Elite"
    if score >= 70:
        return "Strong"
    if score >= 50:
        return "Solid"
    if score >= 30:
        return "Light"
    return "Recovery"
