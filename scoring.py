from __future__ import annotations

"""
Relative Session-Quality Score + Improvement Index for running / riding.

Why two outputs?
- Activity Score (0-100): "How strong / efficient / well-executed was this
  session relative to my recent comparable baseline?"
- Improvement Index (0-100): "Am I improving over time, and by how much?"

This split is intentional. Training-load literature and platform practice both
suggest that a single per-activity score is not enough to infer improvement.
"""

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
import json
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCORE_VERSION = "3.0"

BASELINE_WINDOW_DAYS = 90
CS_LOOKBACK_DAYS = 120
IMPROVEMENT_WINDOW_DAYS = 42

MIN_PERCENTILE_VALUES = 5
MIN_CLASS_BASELINE = 3
MIN_CS_POINTS = 3

TARGET_DISTANCES_M = (400, 800, 1500, 3000, 5000)

RUN_TYPES = {"Run", "TrailRun"}
RIDE_TYPES = {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "E-BikeRide"}

TRAINING_CLASSES = {
    "aerobic": {"easy", "moderate", "long", "recovery"},
    "quality": {"tempo", "intervals", "repetitions", "race", "threshold", "vo2max"},
}

# Class-specific activity-score weights.
# Aerobic sessions can lean more on HR-derived efficiency.
# Quality sessions lean more on pace because average HR is noisier on intervals.
ACTIVITY_WEIGHTS = {
    "aerobic": {
        "pace": 0.40,
        "efficiency": 0.30,
        "execution": 0.20,
        "consistency": 0.10,
    },
    "quality": {
        "pace": 0.50,
        "efficiency": 0.20,
        "execution": 0.20,
        "consistency": 0.10,
    },
}

# Approximate flat-equivalent pace adjustment.
# These are intentionally modest heuristics, not a claim of true GAP equivalence.
RUN_UPHILL_SEC_PER_KM_PER_PCT = 8.0
RUN_DOWNHILL_SEC_PER_KM_PER_PCT = 3.0
RIDE_UPHILL_SEC_PER_KM_PER_PCT = 3.0
RIDE_DOWNHILL_SEC_PER_KM_PER_PCT = 1.0

# Improvement Index blend. This deliberately prioritizes capability signals.
IMPROVEMENT_WEIGHTS = {
    "critical_speed": 0.65,
    "aerobic_efficiency": 0.35,
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_date(date_str: Any) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _activity_dt(activity: Dict[str, Any]) -> datetime:
    dt = _parse_date(activity.get("start_date") or activity.get("start_date_local"))
    if dt is not None:
        return dt
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sport_group(activity: Dict[str, Any]) -> str:
    sport = activity.get("sport_type") or activity.get("type") or "Run"
    return "Ride" if sport in RIDE_TYPES else "Run"


def _broad_training_class(activity: Dict[str, Any]) -> str:
    raw = (activity.get("training_type") or activity.get("workout_type") or "moderate")
    t = str(raw).strip().lower()
    if t in TRAINING_CLASSES["quality"]:
        return "quality"
    return "aerobic"


def _numeric_median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(median(clean))


def _stable_percentile_rank(
    value: Optional[float],
    values: Sequence[Optional[float]],
    *,
    higher_is_better: bool = True,
) -> float:
    """
    Robust percentile with midpoint handling for ties.
    Returns 50 when the baseline is too small.
    """
    if value is None:
        return 50.0

    clean = [float(v) for v in values if v is not None]
    if len(clean) < MIN_PERCENTILE_VALUES:
        return 50.0

    if higher_is_better:
        transformed = sorted(clean)
        x = float(value)
    else:
        transformed = sorted(-v for v in clean)
        x = -float(value)

    left = bisect_left(transformed, x)
    right = bisect_right(transformed, x)
    percentile = ((left + 0.5 * (right - left)) / len(transformed)) * 100.0
    return _clamp(percentile, 0.0, 100.0)


def _band_score(value: float, lo: float, hi: float, soft_margin: float) -> float:
    """
    Score how well a value fits inside a target band.

    - 80-100 inside the band (best near the center)
    - decays outside the band
    - bottoms out at 20 to avoid overconfidence from weak signals
    """
    if soft_margin <= 0:
        return 50.0

    center = (lo + hi) / 2.0
    half_band = max((hi - lo) / 2.0, 1e-9)
    dev = abs(value - center)

    if lo <= value <= hi:
        centeredness = 1.0 - (dev / half_band)
        return 80.0 + 20.0 * _clamp(centeredness, 0.0, 1.0)

    edge_dev = dev - half_band
    if edge_dev >= soft_margin:
        return 20.0

    return 20.0 + 60.0 * (1.0 - edge_dev / soft_margin)


def _trend_score(pct_change: Optional[float], *, cap: float) -> float:
    """
    Map percent change to a 0-100 trend score.
    50 = stable. Positive change improves the score.
    cap controls how much change is needed to get close to the extremes.
    """
    if pct_change is None:
        return 50.0
    scaled = _clamp(pct_change / cap, -1.0, 1.0)
    return _clamp(50.0 + 40.0 * scaled, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Pace / efficiency / consistency helpers
# ---------------------------------------------------------------------------


def _approximate_adjusted_pace(activity: Dict[str, Any], *, is_ride: bool) -> Optional[float]:
    """
    Approximate flat-equivalent pace in sec/km.

    This is intentionally a lightweight hill adjustment, not a claim of true
    grade-adjusted pace. If downhill information is unavailable, only uphill is
    corrected.
    """
    pace = activity.get("pace")
    distance_m = activity.get("distance")
    if not pace or pace <= 0 or not distance_m or distance_m <= 0:
        return None

    gain_m = float(activity.get("total_elevation_gain") or 0.0)
    loss_m = float(
        activity.get("total_elevation_loss")
        or activity.get("elevation_loss")
        or 0.0
    )

    up_pct = (gain_m / distance_m) * 100.0 if gain_m > 0 else 0.0
    down_pct = (loss_m / distance_m) * 100.0 if loss_m > 0 else 0.0

    if is_ride:
        adjusted = float(pace) - up_pct * RIDE_UPHILL_SEC_PER_KM_PER_PCT + down_pct * RIDE_DOWNHILL_SEC_PER_KM_PER_PCT
    else:
        adjusted = float(pace) - up_pct * RUN_UPHILL_SEC_PER_KM_PER_PCT + down_pct * RUN_DOWNHILL_SEC_PER_KM_PER_PCT

    return max(60.0, adjusted)


def _adjusted_speed_m_s(activity: Dict[str, Any], *, is_ride: bool) -> Optional[float]:
    gap = _approximate_adjusted_pace(activity, is_ride=is_ride)
    if gap is None or gap <= 0:
        return None
    return 1000.0 / gap


def _aerobic_efficiency(activity: Dict[str, Any], *, is_ride: bool) -> Optional[float]:
    speed = _adjusted_speed_m_s(activity, is_ride=is_ride)
    hr = activity.get("average_heartrate")
    if speed is None or not hr or hr <= 0:
        return None
    return (speed / float(hr)) * 1000.0


def _extract_stream_blob_from_db(activity_id: Any, db: Any) -> Optional[Any]:
    if db is None or activity_id is None:
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
        raw = row["stream_data"] if not isinstance(row, (list, tuple)) else row[0]
        return json.loads(raw)
    except Exception:
        return None


def _extract_speed_samples(activity: Dict[str, Any], db: Any = None) -> List[float]:
    """
    Try several sources in descending order of quality:
    1) cached Strava-like stream data
    2) embedded velocity_smooth array
    3) 1 km splits / laps converted to segment speed
    """
    activity_id = activity.get("id")
    streams = activity.get("stream_data") or _extract_stream_blob_from_db(activity_id, db)

    velocity_data: Optional[List[float]] = None

    if isinstance(streams, list):
        for item in streams:
            if isinstance(item, dict) and item.get("type") == "velocity_smooth":
                maybe = item.get("data")
                if isinstance(maybe, list):
                    velocity_data = maybe
                    break
    elif isinstance(streams, dict):
        maybe = streams.get("velocity_smooth")
        if isinstance(maybe, dict):
            arr = maybe.get("data")
            if isinstance(arr, list):
                velocity_data = arr
        elif isinstance(maybe, list):
            velocity_data = maybe

    if velocity_data:
        speeds = [float(v) for v in velocity_data if isinstance(v, (int, float)) and v > 0.5]
        if len(speeds) >= 10:
            return speeds

    embedded = activity.get("velocity_smooth")
    if isinstance(embedded, list):
        speeds = [float(v) for v in embedded if isinstance(v, (int, float)) and v > 0.5]
        if len(speeds) >= 10:
            return speeds

    for key in ("splits_metric", "laps", "splits"):
        splits = activity.get(key)
        if not isinstance(splits, list):
            continue
        speeds: List[float] = []
        for s in splits:
            if not isinstance(s, dict):
                continue
            dist = s.get("distance")
            moving = s.get("moving_time") or s.get("elapsed_time")
            if dist and moving and dist > 0 and moving > 0:
                speed = float(dist) / float(moving)
                if speed > 0.5:
                    speeds.append(speed)
        if len(speeds) >= 3:
            return speeds

    return []


def _pace_variability_cv(activity: Dict[str, Any], db: Any = None) -> Optional[float]:
    speeds = _extract_speed_samples(activity, db=db)
    if len(speeds) < 3:
        return None
    mean_speed = sum(speeds) / len(speeds)
    if mean_speed <= 0:
        return None
    variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
    return (variance ** 0.5) / mean_speed


# ---------------------------------------------------------------------------
# Baseline selection
# ---------------------------------------------------------------------------


def _get_prior_baseline(
    all_activities: Sequence[Dict[str, Any]],
    current_activity: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Prior-only baseline to avoid look-ahead bias when scoring historical runs.
    """
    ref_dt = _activity_dt(current_activity)
    cutoff = ref_dt - timedelta(days=BASELINE_WINDOW_DAYS)
    current_id = current_activity.get("id")
    sport_group = _sport_group(current_activity)

    baseline: List[Dict[str, Any]] = []
    for a in all_activities:
        if current_id is not None and a.get("id") == current_id:
            continue
        if _sport_group(a) != sport_group:
            continue
        dt = _activity_dt(a)
        if cutoff <= dt < ref_dt:
            baseline.append(a)
    return baseline


def _split_baseline_by_class(
    baseline: Sequence[Dict[str, Any]],
    broad_class: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    class_baseline = [a for a in baseline if _broad_training_class(a) == broad_class]
    return class_baseline, list(baseline)


# ---------------------------------------------------------------------------
# Critical speed helpers for improvement tracking
# ---------------------------------------------------------------------------


def _normalize_effort_distance(distance_m: Any) -> Optional[int]:
    if distance_m is None:
        return None
    try:
        distance_m = float(distance_m)
    except (TypeError, ValueError):
        return None

    best_target = None
    best_error = None
    for target in TARGET_DISTANCES_M:
        error = abs(distance_m - target) / target
        if best_error is None or error < best_error:
            best_target = target
            best_error = error

    if best_target is not None and best_error is not None and best_error <= 0.08:
        return best_target
    return None


def _best_efforts_from_activity(activity: Dict[str, Any]) -> Dict[int, float]:
    """
    Extract best-effort times from a Strava-like best_efforts field, falling back
    to the whole activity if its distance closely matches one of the target distances.
    """
    best: Dict[int, float] = {}

    efforts = activity.get("best_efforts")
    if isinstance(efforts, list):
        for effort in efforts:
            if not isinstance(effort, dict):
                continue
            target = _normalize_effort_distance(effort.get("distance"))
            if target is None:
                continue
            t = effort.get("moving_time") or effort.get("elapsed_time") or effort.get("time")
            if not t or t <= 0:
                continue
            best[target] = min(best.get(target, float("inf")), float(t))

    whole_distance = _normalize_effort_distance(activity.get("distance"))
    whole_time = activity.get("moving_time") or activity.get("elapsed_time")
    if whole_distance is not None and whole_time and whole_time > 0:
        best[whole_distance] = min(best.get(whole_distance, float("inf")), float(whole_time))

    return best


def _best_effort_table(
    activities: Sequence[Dict[str, Any]],
    *,
    sport_group: str = "Run",
) -> Dict[int, float]:
    table: Dict[int, float] = {}
    for a in activities:
        if _sport_group(a) != sport_group:
            continue
        for dist, t in _best_efforts_from_activity(a).items():
            if t > 0:
                table[dist] = min(table.get(dist, float("inf")), t)
    return table


def _fit_critical_speed(best_times: Dict[int, float]) -> Tuple[Optional[float], Optional[float], int]:
    points = [(float(t), float(d)) for d, t in best_times.items() if t > 0]
    if len(points) < MIN_CS_POINTS:
        return None, None, len(points)

    times = [p[0] for p in points]
    distances = [p[1] for p in points]

    mean_t = sum(times) / len(times)
    mean_d = sum(distances) / len(distances)
    den = sum((t - mean_t) ** 2 for t in times)
    if den <= 0:
        return None, None, len(points)

    cs = sum((t - mean_t) * (d - mean_d) for t, d in points) / den
    d_prime = mean_d - cs * mean_t

    if cs <= 0:
        return None, None, len(points)

    return cs, d_prime, len(points)


def compute_critical_speed(
    activities: Sequence[Dict[str, Any]],
    *,
    sport_group: str = "Run",
) -> Dict[str, Any]:
    best = _best_effort_table(activities, sport_group=sport_group)
    cs, d_prime, point_count = _fit_critical_speed(best)
    return {
        "critical_speed_m_s": round(cs, 4) if cs is not None else None,
        "d_prime_m": round(d_prime, 1) if d_prime is not None else None,
        "best_efforts_used": {int(k): round(v, 1) for k, v in sorted(best.items())},
        "point_count": point_count,
    }


# ---------------------------------------------------------------------------
# Activity Score
# ---------------------------------------------------------------------------


def _execution_quality(
    activity: Dict[str, Any],
    *,
    sport_baseline: Sequence[Dict[str, Any]],
    class_baseline: Sequence[Dict[str, Any]],
    act_gap: Optional[float],
    act_hr: Optional[float],
    cs_m_s: Optional[float],
    is_ride: bool,
) -> Tuple[float, Dict[str, Any]]:
    """
    Conservative intent-alignment score.

    This is intentionally not an "absolute truth" detector. It checks whether
    the session intensity looks plausible for its declared broad class, and it
    uses conservative clamping when it must fall back to sparse/global data.
    """
    broad_class = _broad_training_class(activity)
    fallback_used = False
    method_flags: List[str] = []
    subscores: List[float] = []

    if act_gap is not None:
        speed = 1000.0 / act_gap
    else:
        speed = None

    # 1) Anchor to critical speed when available.
    if speed is not None and cs_m_s is not None and cs_m_s > 0:
        ratio = speed / cs_m_s
        if broad_class == "aerobic":
            subscores.append(_band_score(ratio, 0.68, 0.88, 0.12))
        else:
            subscores.append(_band_score(ratio, 0.80, 1.08, 0.18))
        method_flags.append("critical_speed_anchor")

    class_gaps = [
        _approximate_adjusted_pace(a, is_ride=is_ride)
        for a in class_baseline
    ]
    class_hrs = [
        float(a.get("average_heartrate"))
        for a in class_baseline
        if a.get("average_heartrate")
    ]

    # 2) Compare to prior sessions of the same broad class when possible.
    if act_gap is not None and len([g for g in class_gaps if g is not None]) >= MIN_CLASS_BASELINE:
        pace_pct = _stable_percentile_rank(act_gap, class_gaps, higher_is_better=False) / 100.0
        if broad_class == "aerobic":
            subscores.append(_band_score(pace_pct, 0.20, 0.80, 0.15))
        else:
            subscores.append(_band_score(pace_pct, 0.15, 0.95, 0.10))
        method_flags.append("class_percentile_pace")

    if act_hr is not None and len(class_hrs) >= MIN_CLASS_BASELINE:
        hr_pct = _stable_percentile_rank(act_hr, class_hrs, higher_is_better=True) / 100.0
        if broad_class == "aerobic":
            subscores.append(_band_score(hr_pct, 0.20, 0.80, 0.15))
        else:
            subscores.append(_band_score(hr_pct, 0.15, 0.95, 0.20))
        method_flags.append("class_percentile_hr")

    # 3) Conservative fallback against global sport medians.
    if not subscores:
        fallback_used = True
        global_gaps = [
            _approximate_adjusted_pace(a, is_ride=is_ride)
            for a in sport_baseline
        ]
        global_hrs = [
            float(a.get("average_heartrate"))
            for a in sport_baseline
            if a.get("average_heartrate")
        ]

        med_gap = _numeric_median(global_gaps)
        med_hr = _numeric_median(global_hrs)

        if act_gap is not None and med_gap is not None and med_gap > 0:
            ratio = med_gap / act_gap
            if broad_class == "aerobic":
                subscores.append(_band_score(ratio, 0.90, 1.10, 0.15))
            else:
                subscores.append(_band_score(ratio, 0.98, 1.25, 0.20))
            method_flags.append("global_median_pace")

        if act_hr is not None and med_hr is not None and med_hr > 0:
            ratio = act_hr / med_hr
            if broad_class == "aerobic":
                subscores.append(_band_score(ratio, 0.85, 1.10, 0.15))
            else:
                subscores.append(_band_score(ratio, 0.95, 1.25, 0.20))
            method_flags.append("global_median_hr")

    if not subscores:
        return 50.0, {
            "execution_method": "neutral_no_data",
            "broad_class": broad_class,
            "fallback_used": True,
        }

    score = sum(subscores) / len(subscores)
    if fallback_used:
        score = _clamp(score, 35.0, 65.0)

    return score, {
        "execution_method": ", ".join(method_flags),
        "broad_class": broad_class,
        "class_baseline_size": len(class_baseline),
        "fallback_used": fallback_used,
    }


def _build_confidence(
    *,
    baseline_size: int,
    class_baseline_size: int,
    has_hr: bool,
    has_stream_like: bool,
    used_execution_fallback: bool,
    cs_available: bool,
) -> Tuple[int, str]:
    score = 20.0

    score += min(30.0, baseline_size * 2.0)
    score += min(20.0, class_baseline_size * 3.0)
    if has_hr:
        score += 15.0
    if has_stream_like:
        score += 10.0
    if cs_available:
        score += 10.0
    if used_execution_fallback:
        score -= 15.0

    score = int(round(_clamp(score, 0.0, 100.0)))
    if score >= 80:
        label = "High"
    elif score >= 55:
        label = "Medium"
    else:
        label = "Low"
    return score, label


def score_activity(activity: Dict[str, Any], all_activities: Sequence[Dict[str, Any]], db: Any = None) -> Tuple[int, Dict[str, Any]]:
    """
    Relative Session-Quality Score.

    This is intentionally *not* a fitness or improvement score.
    It answers: "How good was this session relative to recent comparable history?"
    """
    baseline = _get_prior_baseline(all_activities, activity)
    broad_class = _broad_training_class(activity)
    class_baseline, sport_baseline = _split_baseline_by_class(baseline, broad_class)
    comparable_baseline = class_baseline if len(class_baseline) >= MIN_CLASS_BASELINE else sport_baseline

    is_ride = _sport_group(activity) == "Ride"
    weights = dict(ACTIVITY_WEIGHTS[broad_class])

    act_gap = _approximate_adjusted_pace(activity, is_ride=is_ride)
    act_hr = float(activity.get("average_heartrate")) if activity.get("average_heartrate") else None
    has_hr = act_hr is not None and act_hr > 0

    cs_info = compute_critical_speed(baseline, sport_group="Run") if not is_ride else {
        "critical_speed_m_s": None,
        "d_prime_m": None,
        "best_efforts_used": {},
        "point_count": 0,
    }
    cs_m_s = cs_info["critical_speed_m_s"]

    baseline_gaps = [
        _approximate_adjusted_pace(a, is_ride=is_ride)
        for a in comparable_baseline
    ]
    baseline_effs = [
        _aerobic_efficiency(a, is_ride=is_ride)
        for a in comparable_baseline
    ]

    scores: Dict[str, float] = {}
    reasons: Dict[str, str] = {}

    # 1) Pace relative to comparable baseline.
    if act_gap is not None and len([g for g in baseline_gaps if g is not None]) >= MIN_PERCENTILE_VALUES:
        scores["pace"] = _stable_percentile_rank(act_gap, baseline_gaps, higher_is_better=False)
        reasons["pace"] = "comparable_baseline_percentile"
    elif act_gap is not None:
        scores["pace"] = 50.0
        reasons["pace"] = "neutral_sparse_baseline"
    else:
        scores["pace"] = 50.0
        reasons["pace"] = "neutral_missing_pace"

    # 2) Aerobic efficiency.
    act_eff = _aerobic_efficiency(activity, is_ride=is_ride)
    if act_eff is not None and len([e for e in baseline_effs if e is not None]) >= MIN_PERCENTILE_VALUES:
        scores["efficiency"] = _stable_percentile_rank(act_eff, baseline_effs, higher_is_better=True)
        reasons["efficiency"] = "comparable_baseline_percentile"
    elif act_eff is not None:
        scores["efficiency"] = 50.0
        reasons["efficiency"] = "neutral_sparse_baseline"
    else:
        scores["efficiency"] = 50.0
        reasons["efficiency"] = "neutral_missing_hr"
        weights["efficiency"] *= 0.5
        freed = ACTIVITY_WEIGHTS[broad_class]["efficiency"] - weights["efficiency"]
        weights["pace"] += freed * 0.6
        weights["execution"] += freed * 0.4

    # 3) Execution quality.
    exec_score, exec_meta = _execution_quality(
        activity,
        sport_baseline=sport_baseline,
        class_baseline=class_baseline,
        act_gap=act_gap,
        act_hr=act_hr,
        cs_m_s=cs_m_s,
        is_ride=is_ride,
    )
    scores["execution"] = exec_score
    reasons["execution"] = exec_meta.get("execution_method", "unknown")

    # 4) Consistency.
    act_cv = _pace_variability_cv(activity, db=db)
    baseline_cvs = [_pace_variability_cv(a, db=db) for a in comparable_baseline]
    if act_cv is not None and len([cv for cv in baseline_cvs if cv is not None]) >= MIN_PERCENTILE_VALUES:
        scores["consistency"] = _stable_percentile_rank(act_cv, baseline_cvs, higher_is_better=False)
        reasons["consistency"] = "relative_cv_percentile"
    elif act_cv is not None:
        scores["consistency"] = _clamp(100.0 * (1.0 - act_cv / 0.25), 20.0, 100.0)
        reasons["consistency"] = "absolute_cv_fallback"
    else:
        scores["consistency"] = 50.0
        reasons["consistency"] = "neutral_no_stream_or_laps"
        weights["consistency"] *= 0.5
        freed = ACTIVITY_WEIGHTS[broad_class]["consistency"] - weights["consistency"]
        weights["execution"] += freed

    total_weight = sum(weights.values())
    raw_score = sum(scores[k] * weights[k] for k in scores) / total_weight
    final_score = int(round(_clamp(raw_score, 0.0, 100.0)))

    elevation_rate = None
    distance = activity.get("distance") or 0
    gain = activity.get("total_elevation_gain") or 0
    if distance and distance > 500:
        elevation_rate = float(gain) / (float(distance) / 1000.0)

    confidence_score, confidence_label = _build_confidence(
        baseline_size=len(baseline),
        class_baseline_size=len(class_baseline),
        has_hr=has_hr,
        has_stream_like=act_cv is not None,
        used_execution_fallback=bool(exec_meta.get("fallback_used")),
        cs_available=cs_m_s is not None,
    )

    breakdown = {
        "score_version": SCORE_VERSION,
        "score_type": "relative_session_quality",
        "raw_score": round(raw_score, 1),
        "broad_class": broad_class,
        "baseline_size": len(baseline),
        "class_baseline_size": len(class_baseline),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "gap_sec_per_km": round(act_gap, 1) if act_gap is not None else None,
        "aerobic_efficiency_raw": round(act_eff, 3) if act_eff is not None else None,
        "critical_speed_m_s": round(cs_m_s, 3) if cs_m_s is not None else None,
        "critical_speed_points": cs_info.get("point_count"),
        "elevation_difficulty_context": round(elevation_rate, 1) if elevation_rate is not None else None,
        "pace_variability_cv": round(act_cv, 3) if act_cv is not None else None,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "component_reasons": reasons,
        "execution_meta": exec_meta,
        "active_components": {k: round((weights[k] / total_weight) * 100.0, 1) for k in weights},
    }

    return final_score, breakdown


def score_all_activities(activities: Sequence[Dict[str, Any]], db: Any = None) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for a in activities:
        score, breakdown = score_activity(a, activities, db=db)
        copied = dict(a)
        copied["runny_score"] = score
        copied["runny_breakdown"] = breakdown
        result.append(copied)
    return result


# ---------------------------------------------------------------------------
# Improvement Index
# ---------------------------------------------------------------------------


def _latest_date(activities: Sequence[Dict[str, Any]]) -> Optional[datetime]:
    dates = [_activity_dt(a) for a in activities if _activity_dt(a) is not None]
    return max(dates) if dates else None


def _split_recent_previous(
    activities: Sequence[Dict[str, Any]],
    *,
    window_days: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[datetime]]:
    ref_dt = _latest_date(activities)
    if ref_dt is None:
        return [], [], None
    recent_cutoff = ref_dt - timedelta(days=window_days)
    previous_cutoff = ref_dt - timedelta(days=2 * window_days)

    recent: List[Dict[str, Any]] = []
    previous: List[Dict[str, Any]] = []
    for a in activities:
        dt = _activity_dt(a)
        if dt >= recent_cutoff:
            recent.append(a)
        elif previous_cutoff <= dt < recent_cutoff:
            previous.append(a)
    return recent, previous, ref_dt


def _aerobic_efficiency_distribution(activities: Sequence[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for a in activities:
        if _sport_group(a) != "Run":
            continue
        if _broad_training_class(a) != "aerobic":
            continue
        distance = a.get("distance") or 0
        moving = a.get("moving_time") or 0
        if distance < 3000 or moving < 900:
            continue
        eff = _aerobic_efficiency(a, is_ride=False)
        if eff is not None:
            values.append(eff)
    return values


def compute_improvement_index(
    activities: Sequence[Dict[str, Any]],
    *,
    window_days: int = IMPROVEMENT_WINDOW_DAYS,
) -> Dict[str, Any]:
    """
    Improvement Index centered at 50.

    Uses the literature-backed split that best matches the user's goal:
    - critical speed / best-effort capability (primary)
    - aerobic efficiency trend on submaximal aerobic runs (secondary)

    This is intentionally *not* just an EWMA of activity scores.
    """
    empty = {
        "score_version": SCORE_VERSION,
        "improvement_index": 50.0,
        "improvement_pct": 0.0,
        "trend": "stable",
        "window_days": window_days,
        "confidence_score": 0,
        "confidence_label": "Low",
        "critical_speed": {},
        "aerobic_efficiency": {},
    }

    if not activities:
        return empty

    recent, previous, ref_dt = _split_recent_previous(activities, window_days=window_days)
    if ref_dt is None:
        return empty

    recent_runs = [a for a in recent if _sport_group(a) == "Run"]
    previous_runs = [a for a in previous if _sport_group(a) == "Run"]

    recent_cs_info = compute_critical_speed(recent_runs, sport_group="Run")
    previous_cs_info = compute_critical_speed(previous_runs, sport_group="Run")
    recent_cs = recent_cs_info.get("critical_speed_m_s")
    previous_cs = previous_cs_info.get("critical_speed_m_s")

    cs_change = None
    if recent_cs and previous_cs and previous_cs > 0:
        cs_change = (float(recent_cs) - float(previous_cs)) / float(previous_cs)

    recent_eff_values = _aerobic_efficiency_distribution(recent_runs)
    previous_eff_values = _aerobic_efficiency_distribution(previous_runs)
    recent_eff = _numeric_median(recent_eff_values)
    previous_eff = _numeric_median(previous_eff_values)

    eff_change = None
    if recent_eff and previous_eff and previous_eff > 0:
        eff_change = (float(recent_eff) - float(previous_eff)) / float(previous_eff)

    available_weights = {}
    if cs_change is not None:
        available_weights["critical_speed"] = IMPROVEMENT_WEIGHTS["critical_speed"]
    if eff_change is not None:
        available_weights["aerobic_efficiency"] = IMPROVEMENT_WEIGHTS["aerobic_efficiency"]

    if not available_weights:
        return {
            **empty,
            "critical_speed": {
                "recent": recent_cs,
                "previous": previous_cs,
            },
            "aerobic_efficiency": {
                "recent": recent_eff,
                "previous": previous_eff,
            },
        }

    total_w = sum(available_weights.values())
    weighted_change = 0.0
    if cs_change is not None:
        weighted_change += cs_change * available_weights["critical_speed"]
    if eff_change is not None:
        weighted_change += eff_change * available_weights["aerobic_efficiency"]
    weighted_change /= total_w

    cs_score = _trend_score(cs_change, cap=0.06) if cs_change is not None else 50.0
    eff_score = _trend_score(eff_change, cap=0.05) if eff_change is not None else 50.0

    improvement_index = 0.0
    if cs_change is not None:
        improvement_index += cs_score * available_weights["critical_speed"]
    if eff_change is not None:
        improvement_index += eff_score * available_weights["aerobic_efficiency"]
    improvement_index /= total_w
    improvement_index = round(_clamp(improvement_index, 0.0, 100.0), 1)

    if improvement_index >= 57:
        trend = "improving"
    elif improvement_index <= 43:
        trend = "declining"
    else:
        trend = "stable"

    confidence = 20.0
    confidence += min(25.0, len(recent_runs) * 2.0)
    confidence += min(25.0, len(previous_runs) * 2.0)
    confidence += min(15.0, float(recent_cs_info.get("point_count") or 0) * 4.0)
    confidence += min(15.0, float(previous_cs_info.get("point_count") or 0) * 4.0)
    if recent_eff is not None and previous_eff is not None:
        confidence += 10.0
    confidence = int(round(_clamp(confidence, 0.0, 100.0)))

    if confidence >= 80:
        confidence_label = "High"
    elif confidence >= 55:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return {
        "score_version": SCORE_VERSION,
        "improvement_index": improvement_index,
        "improvement_pct": round(weighted_change * 100.0, 2),
        "trend": trend,
        "window_days": window_days,
        "confidence_score": confidence,
        "confidence_label": confidence_label,
        "critical_speed": {
            "recent_m_s": round(recent_cs, 4) if recent_cs is not None else None,
            "previous_m_s": round(previous_cs, 4) if previous_cs is not None else None,
            "pct_change": round(cs_change * 100.0, 2) if cs_change is not None else None,
            "score": round(cs_score, 1) if cs_change is not None else None,
            "recent_points": recent_cs_info.get("point_count"),
            "previous_points": previous_cs_info.get("point_count"),
            "recent_best_efforts": recent_cs_info.get("best_efforts_used"),
            "previous_best_efforts": previous_cs_info.get("best_efforts_used"),
        },
        "aerobic_efficiency": {
            "recent": round(recent_eff, 4) if recent_eff is not None else None,
            "previous": round(previous_eff, 4) if previous_eff is not None else None,
            "pct_change": round(eff_change * 100.0, 2) if eff_change is not None else None,
            "score": round(eff_score, 1) if eff_change is not None else None,
            "recent_sample_size": len(recent_eff_values),
            "previous_sample_size": len(previous_eff_values),
        },
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def get_score_color(score: Optional[float]) -> str:
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


def get_score_label(score: Optional[float]) -> str:
    """Self-referential label for a session-quality score."""
    if score is None:
        return "—"
    if score >= 85:
        return "Exceptional"
    if score >= 70:
        return "Great"
    if score >= 50:
        return "Solid"
    if score >= 30:
        return "Fair"
    return "Sub-par"


def get_improvement_label(index: Optional[float]) -> str:
    if index is None:
        return "—"
    if index >= 57:
        return "Improving"
    if index <= 43:
        return "Declining"
    return "Stable"
