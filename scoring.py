"""
Scoring engine v2 – assigns each activity a score (0-100) relative to the
athlete's recent baseline using efficiency-based metrics.

Philosophy:
- Efficiency over volume (pace & cardiac efficiency, not distance/duration)
- Context-aware (execution quality vs. workout intent)
- Self-referential (percentiles against your own history)
- Graceful degradation (works with missing HR, RPE, training type)
"""
from datetime import datetime, timedelta, timezone
import json

# ─── Configuration ────────────────────────────────────────

BASELINE_WINDOW_DAYS = 90
MIN_ACTIVITIES_FOR_PERCENTILE = 5

# Activity Score component weights (sum = 1.0)
ACTIVITY_WEIGHTS = {
    "pace": 0.30,               # Elevation-adjusted pace (GAP) percentile
    "cardiac_efficiency": 0.25, # speed / avg_hr percentile
    "execution_quality": 0.20,  # How well session matched its intent
    "elevation_difficulty": 0.10,  # Elevation gain per km percentile
    "consistency": 0.15,        # Pace variability (lower = better)
}

# Execution quality: expected pace-zone and HR-zone ranges per training type
# Zones are expressed as percentiles (0.0 = slowest/lowest, 1.0 = fastest/highest)
TRAINING_TYPE_PROFILES = {
    "easy": {
        "pace_zone": (0.00, 0.40),
        "hr_zone": (0.00, 0.40),
        "description": "Low intensity, conversational",
    },
    "moderate": {
        "pace_zone": (0.30, 0.65),
        "hr_zone": (0.30, 0.60),
        "description": "Steady state, comfortable effort",
    },
    "tempo": {
        "pace_zone": (0.60, 0.85),
        "hr_zone": (0.55, 0.80),
        "description": "Comfortably hard, sustainable",
    },
    "long": {
        "pace_zone": (0.00, 0.45),
        "hr_zone": (0.00, 0.45),
        "description": "Extended duration at easy effort",
    },
    "intervals": {
        "pace_zone": (0.75, 1.00),
        "hr_zone": (0.70, 0.95),
        "description": "Hard repeats with recovery",
    },
    "repetitions": {
        "pace_zone": (0.85, 1.00),
        "hr_zone": (0.80, 1.00),
        "description": "Near-max short efforts",
    },
    "race": {
        "pace_zone": (0.80, 1.00),
        "hr_zone": (0.75, 1.00),
        "description": "All-out sustained effort",
    },
}

# GAP adjustment: seconds added per km for each 1% average grade
GAP_SECONDS_PER_PERCENT_GRADE = 12


# ─── Helpers ──────────────────────────────────────────────

def _parse_date(date_str):
    """Parse an ISO date string to a naive datetime."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _get_baseline(all_activities, exclude_id=None):
    """
    Filter activities to the last BASELINE_WINDOW_DAYS,
    excluding the current activity by ID.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=BASELINE_WINDOW_DAYS)
    baseline = []
    for a in all_activities:
        if exclude_id is not None and a.get("id") == exclude_id:
            continue
        dt = _parse_date(a.get("start_date", ""))
        if dt and dt >= cutoff:
            baseline.append(a)
    return baseline


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


def _fractional_rank(value, sorted_values, invert=False):
    """
    Return the fractional position (0.0–1.0) of value within sorted_values.
    Used for execution quality zone matching.
    """
    return _percentile_rank(value, sorted_values, invert) / 100.0


def _elevation_adjusted_pace(pace_sec_per_km, elevation_gain_m, distance_m):
    """
    Approximate Grade Adjusted Pace (GAP).
    Adds time per km based on average grade.
    Returns adjusted pace in sec/km (lower = faster).
    """
    if not pace_sec_per_km or pace_sec_per_km <= 0:
        return pace_sec_per_km
    if not distance_m or distance_m <= 0:
        return pace_sec_per_km

    distance_km = distance_m / 1000.0
    avg_grade_pct = (elevation_gain_m / distance_m) * 100.0 if elevation_gain_m else 0

    # Subtract the grade penalty to get flat-equivalent pace
    # (i.e., a hilly run gets a *better* adjusted pace than the raw pace)
    adjustment = avg_grade_pct * GAP_SECONDS_PER_PERCENT_GRADE
    adjusted = pace_sec_per_km - adjustment
    return max(60, adjusted)  # Floor at 1:00/km to avoid nonsense values


def _cardiac_efficiency(pace_sec_per_km, avg_hr):
    """
    Cardiac efficiency = speed per heartbeat.
    Higher value = more efficient (faster at lower HR).
    Returns value in (m/s) / bpm, scaled for readability.
    """
    if not pace_sec_per_km or pace_sec_per_km <= 0 or not avg_hr or avg_hr <= 0:
        return None
    speed_m_per_s = 1000.0 / pace_sec_per_km
    return (speed_m_per_s / avg_hr) * 1000  # Scale up for nicer numbers


def _execution_quality(activity, baseline_paces, baseline_hrs):
    """
    Score how well the activity matches the expected zone for its
    declared training type.

    Returns 0–100 where 100 = perfectly within the target zone.
    """
    training_type = (activity.get("training_type") or "moderate").lower()
    profile = TRAINING_TYPE_PROFILES.get(training_type, TRAINING_TYPE_PROFILES["moderate"])

    pace = activity.get("pace", 0)
    avg_hr = activity.get("average_heartrate")

    scores = []

    # Pace zone match
    if pace > 0 and baseline_paces:
        pace_frac = _fractional_rank(pace, baseline_paces, invert=True)
        zone_lo, zone_hi = profile["pace_zone"]
        scores.append(_zone_match_score(pace_frac, zone_lo, zone_hi))

    # HR zone match
    if avg_hr and baseline_hrs:
        hr_frac = _fractional_rank(avg_hr, baseline_hrs)
        zone_lo, zone_hi = profile["hr_zone"]
        scores.append(_zone_match_score(hr_frac, zone_lo, zone_hi))

    if not scores:
        return 50.0  # No data to evaluate

    return sum(scores) / len(scores)


def _zone_match_score(actual_frac, zone_lo, zone_hi):
    """
    Score how well `actual_frac` (0–1) fits within [zone_lo, zone_hi].
    Returns 0–100:
    - 100 if perfectly centered in zone
    - Degrades linearly as you move outside the zone
    - Minimum 0 if way outside
    """
    if zone_lo <= actual_frac <= zone_hi:
        # Inside the zone → score 70–100 based on how centered
        zone_center = (zone_lo + zone_hi) / 2.0
        zone_width = zone_hi - zone_lo
        if zone_width == 0:
            return 100.0
        distance_from_center = abs(actual_frac - zone_center)
        centeredness = 1.0 - (distance_from_center / (zone_width / 2.0))
        return 70.0 + 30.0 * centeredness
    else:
        # Outside the zone → score 0–70 based on how far
        if actual_frac < zone_lo:
            distance = zone_lo - actual_frac
        else:
            distance = actual_frac - zone_hi
        # Penalize: 0 if >0.3 away from zone edge
        penalty = min(1.0, distance / 0.3)
        return 70.0 * (1.0 - penalty)


def _pace_consistency_from_streams(activity_id, db=None):
    """
    Compute pace consistency from cached stream data.
    Returns the coefficient of variation (0–1, lower = more consistent),
    or None if stream data is not available.
    """
    if db is None:
        return None

    try:
        row = db.execute(
            "SELECT stream_data FROM streams WHERE activity_id = ?",
            (activity_id,),
        ).fetchone()
    except Exception:
        return None

    if not row:
        return None

    try:
        streams = json.loads(row["stream_data"])
    except (json.JSONDecodeError, TypeError):
        return None

    # Strava returns streams as a list of dicts with "type" and "data" keys
    velocity_data = None
    if isinstance(streams, list):
        for s in streams:
            if s.get("type") == "velocity_smooth":
                velocity_data = s.get("data", [])
                break
    elif isinstance(streams, dict):
        vel_stream = streams.get("velocity_smooth")
        if isinstance(vel_stream, dict):
            velocity_data = vel_stream.get("data", [])
        elif isinstance(vel_stream, list):
            velocity_data = vel_stream

    if not velocity_data or len(velocity_data) < 10:
        return None

    # Filter out zero/near-zero velocities (stopped)
    speeds = [v for v in velocity_data if v > 0.5]
    if len(speeds) < 10:
        return None

    mean = sum(speeds) / len(speeds)
    if mean <= 0:
        return None

    variance = sum((s - mean) ** 2 for s in speeds) / len(speeds)
    std_dev = variance ** 0.5
    cv = std_dev / mean

    return cv


def _consistency_score(activity, baseline, db=None):
    """
    Compute consistency score (0–100).
    Uses stream data if available, otherwise falls back to
    moving_time / elapsed_time ratio.
    """
    activity_id = activity.get("id")

    # Try stream-based consistency first
    cv = _pace_consistency_from_streams(activity_id, db)
    if cv is not None:
        # CV for a typical run: 0.05 (very consistent) to 0.30 (very variable)
        # Map to 0–100: lower CV = higher score
        score = max(0, min(100, (1.0 - (cv / 0.35)) * 100))
        return score, "stream"

    # Fallback: moving_time / elapsed_time ratio
    moving = activity.get("moving_time", 0)
    elapsed = activity.get("elapsed_time", 0)
    if elapsed > 0 and moving > 0:
        ratio = moving / elapsed
        ratios = sorted(
            a.get("moving_time", 0) / a.get("elapsed_time", 1)
            for a in baseline
            if a.get("elapsed_time", 0) > 0 and a.get("moving_time", 0) > 0
        )
        if ratios:
            return _percentile_rank(ratio, ratios), "ratio"

    return 50.0, "fallback"


# ─── Public API ───────────────────────────────────────────

def score_activity(activity: dict, all_activities: list, db=None) -> tuple:
    """
    Score a single activity relative to the baseline.
    Returns (total_score: int, breakdown: dict).

    The breakdown dict contains:
    - Each component's individual score (0–100)
    - Component weights used
    - The final weighted score
    """
    exclude_id = activity.get("id")
    baseline = _get_baseline(all_activities, exclude_id=exclude_id)
    use_percentile = len(baseline) >= MIN_ACTIVITIES_FOR_PERCENTILE

    # ── Build sorted arrays from baseline ──
    baseline_paces = sorted(
        _elevation_adjusted_pace(
            a.get("pace", 0),
            a.get("total_elevation_gain", 0),
            a.get("distance", 0),
        )
        for a in baseline
        if a.get("pace", 0) > 0
    )
    baseline_efficiencies = sorted(
        e for e in (
            _cardiac_efficiency(a.get("pace", 0), a.get("average_heartrate"))
            for a in baseline
        )
        if e is not None
    )
    baseline_elev_rates = sorted(
        (a.get("total_elevation_gain", 0) / (a.get("distance", 0) / 1000.0))
        for a in baseline
        if a.get("distance", 0) > 500  # At least 500m
    )

    breakdown = {}
    weighted_sum = 0.0
    active_weight = 0.0

    # ── 1. Pace (GAP-adjusted, inverted: faster = higher score) ──
    pace = activity.get("pace", 0)
    if pace > 0:
        gap = _elevation_adjusted_pace(
            pace,
            activity.get("total_elevation_gain", 0),
            activity.get("distance", 0),
        )
        if use_percentile and baseline_paces:
            p = _percentile_rank(gap, baseline_paces, invert=True)
        else:
            # Fallback: ratio to 6:00/km (360s)
            p = min(100, max(0, (360 / gap) * 50))
        breakdown["pace"] = round(p, 1)
        breakdown["gap_sec_per_km"] = round(gap, 1)
        weighted_sum += p * ACTIVITY_WEIGHTS["pace"]
        active_weight += ACTIVITY_WEIGHTS["pace"]

    # ── 2. Cardiac efficiency (speed / HR, higher = better) ──
    avg_hr = activity.get("average_heartrate")
    ce = _cardiac_efficiency(pace, avg_hr)
    if ce is not None:
        if use_percentile and baseline_efficiencies:
            p = _percentile_rank(ce, baseline_efficiencies)
        else:
            # Fallback: ratio to a typical efficiency
            p = min(100, max(0, (ce / 20.0) * 50))
        breakdown["cardiac_efficiency"] = round(p, 1)
        breakdown["cardiac_efficiency_raw"] = round(ce, 2)
        weighted_sum += p * ACTIVITY_WEIGHTS["cardiac_efficiency"]
        active_weight += ACTIVITY_WEIGHTS["cardiac_efficiency"]

    # ── 3. Execution quality (zone match) ──
    if use_percentile and baseline_paces:
        baseline_hrs = sorted(
            a["average_heartrate"]
            for a in baseline
            if a.get("average_heartrate")
        )
        eq = _execution_quality(activity, baseline_paces, baseline_hrs)
        training_type = (activity.get("training_type") or "moderate").lower()
        breakdown["execution_quality"] = round(eq, 1)
        breakdown["training_type"] = training_type
        weighted_sum += eq * ACTIVITY_WEIGHTS["execution_quality"]
        active_weight += ACTIVITY_WEIGHTS["execution_quality"]

    # ── 4. Elevation difficulty (elev gain per km, higher = harder) ──
    distance = activity.get("distance", 0)
    elev = activity.get("total_elevation_gain", 0)
    if distance > 500:
        elev_rate = elev / (distance / 1000.0)
        if use_percentile and baseline_elev_rates:
            p = _percentile_rank(elev_rate, baseline_elev_rates)
        else:
            # Fallback: ratio to 10m/km
            p = min(100, max(0, (elev_rate / 10.0) * 50))
        breakdown["elevation_difficulty"] = round(p, 1)
        breakdown["elev_per_km"] = round(elev_rate, 1)
        weighted_sum += p * ACTIVITY_WEIGHTS["elevation_difficulty"]
        active_weight += ACTIVITY_WEIGHTS["elevation_difficulty"]

    # ── 5. Consistency ──
    cons_score, cons_source = _consistency_score(activity, baseline, db=db)
    breakdown["consistency"] = round(cons_score, 1)
    breakdown["consistency_source"] = cons_source
    weighted_sum += cons_score * ACTIVITY_WEIGHTS["consistency"]
    active_weight += ACTIVITY_WEIGHTS["consistency"]

    # ── Final score ──
    raw_score = (weighted_sum / active_weight) if active_weight > 0 else 50.0
    final_score = int(min(100, max(0, round(raw_score))))

    breakdown["raw_score"] = round(raw_score, 1)
    breakdown["active_components"] = {
        k: round(ACTIVITY_WEIGHTS[k] / active_weight * 100, 1)
        for k in ACTIVITY_WEIGHTS
        if k in breakdown
    }

    return final_score, breakdown


def score_all_activities(activities: list, db=None) -> list:
    """Score all activities in bulk, returning them with runny_score set."""
    result = []
    for a in activities:
        score, _ = score_activity(a, activities, db=db)
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
