from __future__ import annotations

"""
Improvement Index v3.

Built specifically for runners who want to understand if they are improving,
not just whether they trained hard.

Main idea:
- Do NOT use EWMA of activity scores as the main improvement signal.
- Use capability and aerobic development signals first.
- Keep EWMA only for chart / backward compatibility.

Primary signals:
1) Critical Speed trend from recent best efforts.
2) Aerobic efficiency trend on comparable aerobic runs.
3) Adjusted aerobic pace trend as a conservative fallback.

This module is designed to work with the current scoring.py file.
"""

from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scoring import (
    IMPROVEMENT_WINDOW_DAYS as DEFAULT_TREND_WINDOW_DAYS,
    SCORE_VERSION as ACTIVITY_SCORE_VERSION,
    _aerobic_efficiency,
    _approximate_adjusted_pace,
    _broad_training_class,
    _clamp,
    _trend_score,
    compute_critical_speed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMPROVEMENT_VERSION = f"{ACTIVITY_SCORE_VERSION}.improvement"
EWMA_ALPHA = 0.10  # chart smoothing only
MIN_ACTIVITIES_PER_WINDOW = 3
MIN_AEROBIC_RUNS_PER_WINDOW = 2


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: Any) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _safe_activity_dt(activity: Dict[str, Any]) -> Optional[datetime]:
    return _parse_date(activity.get("start_date") or activity.get("start_date_local"))


def _numeric_median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(median(clean))


def _sport_group(activity: Dict[str, Any]) -> str:
    sport = activity.get("sport_type") or activity.get("type") or "Run"
    if sport in {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "E-BikeRide"}:
        return "Ride"
    return "Run"


def _run_activities(activities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [a for a in activities if _sport_group(a) == "Run"]


def _aerobic_runs(activities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Restrict to comparable aerobic runs where submaximal signals are more stable.
    """
    aerobic: List[Dict[str, Any]] = []
    for a in _run_activities(activities):
        if _broad_training_class(a) != "aerobic":
            continue
        distance = float(a.get("distance") or 0.0)
        moving = float(a.get("moving_time") or 0.0)
        if distance < 3000 or moving < 900:
            continue
        aerobic.append(a)
    return aerobic


def _latest_date(activities: Sequence[Dict[str, Any]]) -> Optional[datetime]:
    dates = [dt for dt in (_safe_activity_dt(a) for a in activities) if dt is not None]
    return max(dates) if dates else None


def _sorted_activities(activities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(activities, key=lambda a: _safe_activity_dt(a) or datetime.min)


def _split_windows(
    activities: Sequence[Dict[str, Any]],
    window_days: int = DEFAULT_TREND_WINDOW_DAYS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[datetime], Optional[datetime]]:
    """
    Split activities into two consecutive windows using the latest valid
    activity date as the reference point.

    Returns: recent, previous, reference_date, recent_cutoff
    """
    ref_dt = _latest_date(activities)
    if ref_dt is None:
        return [], [], None, None

    cutoff_recent = ref_dt - timedelta(days=window_days)
    cutoff_previous = ref_dt - timedelta(days=window_days * 2)

    recent: List[Dict[str, Any]] = []
    previous: List[Dict[str, Any]] = []

    for a in activities:
        dt = _safe_activity_dt(a)
        if dt is None:
            continue
        if dt >= cutoff_recent:
            recent.append(a)
        elif cutoff_previous <= dt < cutoff_recent:
            previous.append(a)

    return recent, previous, ref_dt, cutoff_recent


# ---------------------------------------------------------------------------
# EWMA history (charts / backward compatibility only)
# ---------------------------------------------------------------------------


def compute_ewma(scores: Sequence[float], alpha: float = EWMA_ALPHA) -> List[float]:
    if not scores:
        return []

    ewma = [float(scores[0])]
    for i in range(1, len(scores)):
        ewma.append(alpha * float(scores[i]) + (1.0 - alpha) * ewma[i - 1])
    return ewma


def _build_ewma_history(
    activities: Sequence[Dict[str, Any]],
    recent_cutoff: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    sorted_acts = _sorted_activities(activities)
    scores = [float(a.get("runny_score", 0) or 0) for a in sorted_acts]
    ewma = compute_ewma(scores)

    history: List[Dict[str, Any]] = []
    previous_ewma = 0.0
    for i, a in enumerate(sorted_acts):
        dt = _safe_activity_dt(a)
        history.append(
            {
                "date": a.get("start_date", ""),
                "name": a.get("name", ""),
                "score": float(a.get("runny_score", 0) or 0),
                "ewma": round(ewma[i], 1),
                "pace": a.get("pace", 0),
                "moving_time": a.get("moving_time", 0),
                "distance": a.get("distance", 0),
                "average_heartrate": a.get("average_heartrate"),
                "max_heartrate": a.get("max_heartrate"),
            }
        )
        if recent_cutoff is not None and dt is not None and dt < recent_cutoff:
            previous_ewma = ewma[i]

    current_ewma = ewma[-1] if ewma else 0.0
    ewma_pct = ((current_ewma - previous_ewma) / previous_ewma * 100.0) if previous_ewma > 0 else 0.0
    return history, round(current_ewma, 1), round(previous_ewma, 1), round(ewma_pct, 2)


# ---------------------------------------------------------------------------
# Window-specific running signals
# ---------------------------------------------------------------------------


def _median_adjusted_pace(aerobic_runs: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [_approximate_adjusted_pace(a, is_ride=False) for a in aerobic_runs]
    return _numeric_median(values)


def _efficiency_values(aerobic_runs: Sequence[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for a in aerobic_runs:
        eff = _aerobic_efficiency(a, is_ride=False)
        if eff is not None:
            values.append(eff)
    return values


def _median_efficiency(aerobic_runs: Sequence[Dict[str, Any]]) -> Optional[float]:
    return _numeric_median(_efficiency_values(aerobic_runs))


def _pct_change(
    recent_value: Optional[float],
    previous_value: Optional[float],
    *,
    higher_is_better: bool,
) -> Optional[float]:
    if recent_value is None or previous_value is None or previous_value <= 0:
        return None
    if higher_is_better:
        return (float(recent_value) - float(previous_value)) / float(previous_value)
    return (float(previous_value) - float(recent_value)) / float(previous_value)


def _run_volume_ratio(recent_runs: Sequence[Dict[str, Any]], previous_runs: Sequence[Dict[str, Any]]) -> Optional[float]:
    recent_dist = sum(float(a.get("distance") or 0.0) for a in recent_runs)
    previous_dist = sum(float(a.get("distance") or 0.0) for a in previous_runs)
    if previous_dist <= 0:
        return None
    return recent_dist / previous_dist


def _choose_scoring_path(
    cs_change: Optional[float],
    eff_change: Optional[float],
    pace_change: Optional[float],
) -> Tuple[str, Dict[str, float]]:
    """
    Decide which signals drive the improvement index.
    Priority:
    1) critical speed + aerobic efficiency
    2) critical speed + adjusted pace
    3) aerobic efficiency + adjusted pace
    4) best available single signal
    """
    if cs_change is not None and eff_change is not None:
        return "critical_speed+aerobic_efficiency", {
            "critical_speed": 0.65,
            "aerobic_efficiency": 0.35,
        }
    if cs_change is not None and pace_change is not None:
        return "critical_speed+adjusted_pace", {
            "critical_speed": 0.75,
            "adjusted_pace": 0.25,
        }
    if eff_change is not None and pace_change is not None:
        return "aerobic_efficiency+adjusted_pace", {
            "aerobic_efficiency": 0.60,
            "adjusted_pace": 0.40,
        }
    if cs_change is not None:
        return "critical_speed_only", {"critical_speed": 1.0}
    if eff_change is not None:
        return "aerobic_efficiency_only", {"aerobic_efficiency": 1.0}
    if pace_change is not None:
        return "adjusted_pace_only", {"adjusted_pace": 1.0}
    return "neutral", {}


def _build_confidence(
    *,
    recent_run_count: int,
    previous_run_count: int,
    recent_aerobic_count: int,
    previous_aerobic_count: int,
    recent_cs_points: int,
    previous_cs_points: int,
    used_path: str,
) -> Tuple[int, str]:
    score = 15.0
    score += min(20.0, recent_run_count * 1.5)
    score += min(20.0, previous_run_count * 1.5)
    score += min(15.0, recent_aerobic_count * 2.0)
    score += min(15.0, previous_aerobic_count * 2.0)
    score += min(7.5, recent_cs_points * 2.5)
    score += min(7.5, previous_cs_points * 2.5)

    if used_path == "neutral":
        score -= 25.0
    elif used_path.endswith("_only"):
        score -= 10.0
    elif "adjusted_pace" in used_path and "critical_speed" not in used_path:
        score -= 5.0

    score = int(round(_clamp(score, 0.0, 100.0)))
    if score >= 80:
        return score, "High"
    if score >= 55:
        return score, "Medium"
    return score, "Low"


def _trend_from_index(index: float) -> str:
    if index >= 57:
        return "improving"
    if index <= 43:
        return "declining"
    return "flat"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_improvement(
    activities: Sequence[Dict[str, Any]],
    window_days: int = DEFAULT_TREND_WINDOW_DAYS,
) -> Dict[str, Any]:
    """
    Compute improvement from two consecutive windows.

    Keeps the old payload shape where useful, but the scoring logic is rebuilt:
    - not based on EWMA of scores
    - centered on capability + efficiency trends
    - uses EWMA only for chart compatibility / sparse-data fallback
    """
    recent, previous, ref_dt, recent_cutoff = _split_windows(activities, window_days=window_days)
    ewma_history, current_ewma, previous_ewma, ewma_pct = _build_ewma_history(activities, recent_cutoff)

    empty = {
        "score_version": IMPROVEMENT_VERSION,
        "improvement_index": 50.0,
        "trend": "flat",
        "window_days": window_days,
        "pace_trend": None,
        "efficiency_trend": None,
        "volume_trend": None,
        "breakdown": {},
        "ewma_history": ewma_history,
        "current_ewma": current_ewma,
        "previous_ewma": previous_ewma,
        "improvement_pct": 0.0,
        "confidence_score": 0,
        "confidence_label": "Low",
        "scoring_path": "neutral",
        "critical_speed": {},
        "aerobic_efficiency": {},
        "adjusted_pace": {},
        "volume": {},
    }

    if not activities or ref_dt is None:
        return empty

    recent_runs = _run_activities(recent)
    previous_runs = _run_activities(previous)
    recent_aerobic = _aerobic_runs(recent)
    previous_aerobic = _aerobic_runs(previous)

    recent_cs_info = compute_critical_speed(recent_runs, sport_group="Run")
    previous_cs_info = compute_critical_speed(previous_runs, sport_group="Run")
    recent_cs = recent_cs_info.get("critical_speed_m_s")
    previous_cs = previous_cs_info.get("critical_speed_m_s")
    cs_change = _pct_change(recent_cs, previous_cs, higher_is_better=True)

    recent_eff = _median_efficiency(recent_aerobic)
    previous_eff = _median_efficiency(previous_aerobic)
    eff_change = _pct_change(recent_eff, previous_eff, higher_is_better=True)

    recent_pace = _median_adjusted_pace(recent_aerobic)
    previous_pace = _median_adjusted_pace(previous_aerobic)
    pace_change = _pct_change(recent_pace, previous_pace, higher_is_better=False)

    volume_ratio = _run_volume_ratio(recent_runs, previous_runs)

    # Compatibility fields matching the old API shape.
    pace_trend = None
    if recent_pace is not None and previous_pace is not None:
        pace_trend = round(float(previous_pace) - float(recent_pace), 1)

    efficiency_trend = None
    if recent_eff is not None and previous_eff is not None:
        efficiency_trend = round(float(recent_eff) - float(previous_eff), 3)

    volume_trend = round(volume_ratio, 2) if volume_ratio is not None else None

    scoring_path, weights = _choose_scoring_path(cs_change, eff_change, pace_change)

    breakdown: Dict[str, Any] = {
        "reference_date": ref_dt.isoformat(),
        "scoring_path": scoring_path,
        "window_counts": {
            "recent_total": len(recent),
            "previous_total": len(previous),
            "recent_runs": len(recent_runs),
            "previous_runs": len(previous_runs),
            "recent_aerobic_runs": len(recent_aerobic),
            "previous_aerobic_runs": len(previous_aerobic),
        },
        "critical_speed": {
            "recent_m_s": round(recent_cs, 4) if recent_cs is not None else None,
            "previous_m_s": round(previous_cs, 4) if previous_cs is not None else None,
            "pct_change": round(cs_change * 100.0, 2) if cs_change is not None else None,
            "recent_points": recent_cs_info.get("point_count"),
            "previous_points": previous_cs_info.get("point_count"),
            "recent_best_efforts": recent_cs_info.get("best_efforts_used"),
            "previous_best_efforts": previous_cs_info.get("best_efforts_used"),
        },
        "aerobic_efficiency": {
            "recent": round(recent_eff, 4) if recent_eff is not None else None,
            "previous": round(previous_eff, 4) if previous_eff is not None else None,
            "pct_change": round(eff_change * 100.0, 2) if eff_change is not None else None,
            "recent_sample_size": len(_efficiency_values(recent_aerobic)),
            "previous_sample_size": len(_efficiency_values(previous_aerobic)),
        },
        "adjusted_pace": {
            "recent_sec_per_km": round(recent_pace, 1) if recent_pace is not None else None,
            "previous_sec_per_km": round(previous_pace, 1) if previous_pace is not None else None,
            "delta_sec_per_km": pace_trend,
            "pct_change": round(pace_change * 100.0, 2) if pace_change is not None else None,
        },
        "volume": {
            "recent_km": round(sum(float(a.get("distance") or 0.0) for a in recent_runs) / 1000.0, 1),
            "previous_km": round(sum(float(a.get("distance") or 0.0) for a in previous_runs) / 1000.0, 1),
            "ratio": volume_trend,
            "used_in_score": False,
        },
        "ewma": {
            "current": current_ewma,
            "previous": previous_ewma,
            "pct_change": ewma_pct,
            "note": "Chart / backward-compatibility only. Not a primary improvement signal.",
        },
    }

    # Primary scoring path.
    if weights:
        component_scores: Dict[str, float] = {}
        weighted_change = 0.0
        total_weight = 0.0

        if "critical_speed" in weights and cs_change is not None:
            component_scores["critical_speed"] = _trend_score(cs_change, cap=0.06)
            weighted_change += cs_change * weights["critical_speed"]
            total_weight += weights["critical_speed"]

        if "aerobic_efficiency" in weights and eff_change is not None:
            component_scores["aerobic_efficiency"] = _trend_score(eff_change, cap=0.05)
            weighted_change += eff_change * weights["aerobic_efficiency"]
            total_weight += weights["aerobic_efficiency"]

        if "adjusted_pace" in weights and pace_change is not None:
            component_scores["adjusted_pace"] = _trend_score(pace_change, cap=0.04)
            weighted_change += pace_change * weights["adjusted_pace"]
            total_weight += weights["adjusted_pace"]

        if total_weight > 0:
            improvement_index = round(
                sum(component_scores[k] * weights[k] for k in component_scores) / total_weight,
                1,
            )
            improvement_pct = round((weighted_change / total_weight) * 100.0, 2)
            trend = _trend_from_index(improvement_index)
        else:
            improvement_index = 50.0
            improvement_pct = 0.0
            trend = "flat"

        breakdown["components"] = {k: round(v, 1) for k, v in component_scores.items()}
        breakdown["active_weights"] = {k: round(v * 100.0, 1) for k, v in weights.items()}
    else:
        # Last-resort fallback: very weak EWMA-derived estimate.
        if len(ewma_history) >= 4 and previous_ewma > 0:
            fallback_pct = ewma_pct / 100.0
            improvement_index = round(_trend_score(fallback_pct, cap=0.08), 1)
            improvement_pct = round(ewma_pct, 2)
            trend = _trend_from_index(improvement_index)
            scoring_path = "ewma_score_fallback"
            breakdown["fallback_note"] = (
                "Insufficient critical-speed / efficiency / aerobic-pace data; "
                "using EWMA of activity scores as a weak fallback."
            )
            breakdown["components"] = {"ewma_score": improvement_index}
            breakdown["active_weights"] = {"ewma_score": 100.0}
        else:
            improvement_index = 50.0
            improvement_pct = 0.0
            trend = "flat"
            scoring_path = "neutral"
            breakdown["fallback_note"] = "Not enough comparable data to infer improvement confidently."
            breakdown["components"] = {}
            breakdown["active_weights"] = {}

    confidence_score, confidence_label = _build_confidence(
        recent_run_count=len(recent_runs),
        previous_run_count=len(previous_runs),
        recent_aerobic_count=len(recent_aerobic),
        previous_aerobic_count=len(previous_aerobic),
        recent_cs_points=int(recent_cs_info.get("point_count") or 0),
        previous_cs_points=int(previous_cs_info.get("point_count") or 0),
        used_path=scoring_path,
    )

    if len(recent_runs) < MIN_ACTIVITIES_PER_WINDOW or len(previous_runs) < MIN_ACTIVITIES_PER_WINDOW:
        breakdown["note"] = (
            f"Need at least {MIN_ACTIVITIES_PER_WINDOW} runs in each window for stronger confidence."
        )
        confidence_score = min(confidence_score, 40)
        confidence_label = "Low"

    return {
        "score_version": IMPROVEMENT_VERSION,
        "improvement_index": float(improvement_index),
        "trend": trend,
        "window_days": window_days,
        "pace_trend": pace_trend,
        "efficiency_trend": efficiency_trend,
        "volume_trend": volume_trend,
        "breakdown": breakdown,
        "ewma_history": ewma_history,
        "current_ewma": current_ewma,
        "previous_ewma": previous_ewma,
        "improvement_pct": float(improvement_pct),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "scoring_path": scoring_path,
        "critical_speed": {
            "recent_m_s": round(recent_cs, 4) if recent_cs is not None else None,
            "previous_m_s": round(previous_cs, 4) if previous_cs is not None else None,
            "pct_change": round(cs_change * 100.0, 2) if cs_change is not None else None,
            "score": round(_trend_score(cs_change, cap=0.06), 1) if cs_change is not None else None,
            "recent_points": recent_cs_info.get("point_count"),
            "previous_points": previous_cs_info.get("point_count"),
            "recent_best_efforts": recent_cs_info.get("best_efforts_used"),
            "previous_best_efforts": previous_cs_info.get("best_efforts_used"),
        },
        "aerobic_efficiency": {
            "recent": round(recent_eff, 4) if recent_eff is not None else None,
            "previous": round(previous_eff, 4) if previous_eff is not None else None,
            "pct_change": round(eff_change * 100.0, 2) if eff_change is not None else None,
            "score": round(_trend_score(eff_change, cap=0.05), 1) if eff_change is not None else None,
            "recent_sample_size": len(_efficiency_values(recent_aerobic)),
            "previous_sample_size": len(_efficiency_values(previous_aerobic)),
        },
        "adjusted_pace": {
            "recent_sec_per_km": round(recent_pace, 1) if recent_pace is not None else None,
            "previous_sec_per_km": round(previous_pace, 1) if previous_pace is not None else None,
            "delta_sec_per_km": pace_trend,
            "pct_change": round(pace_change * 100.0, 2) if pace_change is not None else None,
            "score": round(_trend_score(pace_change, cap=0.04), 1) if pace_change is not None else None,
            "recent_sample_size": len([_approximate_adjusted_pace(a, is_ride=False) for a in recent_aerobic if _approximate_adjusted_pace(a, is_ride=False) is not None]),
            "previous_sample_size": len([_approximate_adjusted_pace(a, is_ride=False) for a in previous_aerobic if _approximate_adjusted_pace(a, is_ride=False) is not None]),
        },
        "volume": {
            "recent_km": round(sum(float(a.get("distance") or 0.0) for a in recent_runs) / 1000.0, 1),
            "previous_km": round(sum(float(a.get("distance") or 0.0) for a in previous_runs) / 1000.0, 1),
            "ratio": volume_trend,
            "used_in_score": False,
        },
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def get_trend_color(trend: str) -> str:
    """Return a CSS color for the trend."""
    return {
        "improving": "#00e676",
        "declining": "#f44336",
        "flat": "#ffab00",
        "stable": "#ffab00",
    }.get(trend, "#ffab00")


def get_trend_icon(trend: str) -> str:
    """Return an emoji icon for the trend."""
    return {
        "improving": "📈",
        "declining": "📉",
        "flat": "➡️",
        "stable": "➡️",
    }.get(trend, "➡️")
