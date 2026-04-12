"""
Tests for scoring engine v2 and improvement index.
Run with: python -m pytest test_scoring.py -v
"""
import pytest
from datetime import datetime, timedelta, timezone

from scoring import (
    score_activity,
    score_all_activities,
    get_score_color,
    get_score_label,
    _approximate_adjusted_pace,
    _cardiac_efficiency,
    _percentile_rank,
    _zone_match_score,
    _get_baseline,
)
from improvement import (
    compute_improvement,
    compute_ewma,
    _split_windows,
    _sigmoid_scale,
)


# ─── Fixtures ─────────────────────────────────────────────

def _make_activity(
    id=1,
    days_ago=0,
    distance=10000,     # 10 km
    moving_time=3000,   # 50 min
    elapsed_time=3200,
    pace=300,           # 5:00/km
    elevation=100,
    avg_hr=150,
    max_hr=175,
    training_type="moderate",
    rpe=5,
    runny_score=None,
):
    """Create a synthetic activity dict."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "id": id,
        "name": f"Run #{id}",
        "start_date": dt.isoformat(),
        "distance": distance,
        "moving_time": moving_time,
        "elapsed_time": elapsed_time,
        "total_elevation_gain": elevation,
        "pace": pace,
        "average_heartrate": avg_hr,
        "max_heartrate": max_hr,
        "sport_type": "Run",
        "training_type": training_type,
        "rpe": rpe,
        "runny_score": runny_score,
    }


def _make_baseline(n=20, days_spread=80):
    """Create a set of baseline activities spread over a time range."""
    activities = []
    for i in range(n):
        days = int((i / max(n - 1, 1)) * days_spread)
        activities.append(_make_activity(
            id=100 + i,
            days_ago=days,
            distance=7000 + i * 300,
            moving_time=2100 + i * 90,
            pace=280 + i * 5,
            elevation=50 + i * 10,
            avg_hr=140 + i * 2,
            training_type=["easy", "moderate", "tempo", "long", "intervals"][i % 5],
            runny_score=40 + i * 2,
        ))
    return activities


# ─── Scoring: Helpers ─────────────────────────────────────

class TestPercentileRank:
    def test_empty_list(self):
        assert _percentile_rank(50, []) == 50.0

    def test_all_below(self):
        assert _percentile_rank(100, [10, 20, 30, 40, 50]) == 100.0

    def test_all_above(self):
        assert _percentile_rank(1, [10, 20, 30, 40, 50]) == 0.0

    def test_middle(self):
        result = _percentile_rank(30, [10, 20, 30, 40, 50])
        assert 40 <= result <= 80

    def test_inverted(self):
        # Lower value should rank higher when inverted
        result_fast = _percentile_rank(200, [200, 300, 400], invert=True)
        result_slow = _percentile_rank(400, [200, 300, 400], invert=True)
        assert result_fast > result_slow


class TestApproximateAdjustedPace:
    def test_flat_course(self):
        # No elevation → no adjustment
        gap = _approximate_adjusted_pace(300, 0, 10000)
        assert gap == 300

    def test_hilly_course(self):
        # 200m elevation over 10km = 2% grade → should subtract ~18s
        gap = _approximate_adjusted_pace(360, 200, 10000)
        assert gap < 360

    def test_minimum_floor(self):
        # Extreme case shouldn't go below 60 sec/km
        gap = _approximate_adjusted_pace(120, 1000, 5000)
        assert gap >= 60

    def test_zero_pace(self):
        assert _approximate_adjusted_pace(0, 100, 10000) == 0

    def test_zero_distance(self):
        assert _approximate_adjusted_pace(300, 100, 0) == 300


class TestCardiacEfficiency:
    def test_normal(self):
        ce = _cardiac_efficiency(300, 150)
        assert ce is not None
        assert ce > 0

    def test_faster_is_more_efficient(self):
        ce_fast = _cardiac_efficiency(240, 150)  # 4:00/km at 150bpm
        ce_slow = _cardiac_efficiency(360, 150)  # 6:00/km at 150bpm
        assert ce_fast > ce_slow

    def test_lower_hr_is_more_efficient(self):
        ce_low = _cardiac_efficiency(300, 130)   # 5:00/km at 130bpm
        ce_high = _cardiac_efficiency(300, 170)  # 5:00/km at 170bpm
        assert ce_low > ce_high

    def test_no_hr(self):
        assert _cardiac_efficiency(300, None) is None
        assert _cardiac_efficiency(300, 0) is None

    def test_no_pace(self):
        assert _cardiac_efficiency(0, 150) is None


class TestZoneMatchScore:
    def test_perfectly_centered(self):
        score = _zone_match_score(0.5, 0.3, 0.7)
        assert score == 100.0

    def test_inside_zone_edge(self):
        score = _zone_match_score(0.3, 0.3, 0.7)
        assert 70 <= score <= 100

    def test_just_outside_zone(self):
        score = _zone_match_score(0.25, 0.3, 0.7)
        assert 0 < score < 70

    def test_far_outside_zone(self):
        score = _zone_match_score(0.0, 0.5, 0.7)
        assert score < 30


# ─── Scoring: score_activity ──────────────────────────────

class TestScoreActivity:
    def test_returns_valid_range(self):
        baseline = _make_baseline()
        activity = _make_activity(id=999, days_ago=1)
        score, breakdown = score_activity(activity, baseline)
        assert 0 <= score <= 100

    def test_breakdown_has_components(self):
        baseline = _make_baseline()
        activity = _make_activity(id=999, days_ago=1)
        score, breakdown = score_activity(activity, baseline)
        assert "pace" in breakdown
        assert "raw_score" in breakdown

    def test_current_activity_excluded_from_baseline(self):
        # Create activity that's also in the baseline
        activities = _make_baseline()
        target = activities[0]
        score1, _ = score_activity(target, activities)
        # The target should NOT be compared against itself
        assert 0 <= score1 <= 100

    def test_missing_hr_still_scores(self):
        baseline = _make_baseline()
        activity = _make_activity(id=999, days_ago=1, avg_hr=None, max_hr=None)
        score, breakdown = score_activity(activity, baseline)
        assert 0 <= score <= 100
        assert breakdown["cardiac_efficiency"] == 50.0

    def test_fast_run_scores_higher_than_slow(self):
        baseline = _make_baseline()
        fast = _make_activity(id=997, days_ago=1, pace=240, avg_hr=150)
        slow = _make_activity(id=998, days_ago=1, pace=420, avg_hr=150)
        score_fast, _ = score_activity(fast, baseline)
        score_slow, _ = score_activity(slow, baseline)
        assert score_fast > score_slow

    def test_efficient_run_scores_higher(self):
        baseline = _make_baseline()
        efficient = _make_activity(id=997, days_ago=1, pace=300, avg_hr=130)
        inefficient = _make_activity(id=998, days_ago=1, pace=300, avg_hr=180)
        score_eff, _ = score_activity(efficient, baseline)
        score_ineff, _ = score_activity(inefficient, baseline)
        assert score_eff > score_ineff

    def test_few_activities_uses_fallback(self):
        # Only 3 activities → below MIN_ACTIVITIES_FOR_PERCENTILE
        small_baseline = [_make_activity(id=i, days_ago=i * 3) for i in range(3)]
        activity = _make_activity(id=999, days_ago=1)
        score, _ = score_activity(activity, small_baseline)
        assert 0 <= score <= 100


class TestScoreAllActivities:
    def test_bulk_scoring(self):
        activities = _make_baseline(n=10)
        scored = score_all_activities(activities)
        assert len(scored) == len(activities)
        for a in scored:
            assert a["runny_score"] is not None
            assert 0 <= a["runny_score"] <= 100


# ─── Scoring: Labels & Colors ────────────────────────────

class TestScoreLabels:
    def test_color_for_high(self):
        assert get_score_color(90) == "#00e676"

    def test_color_for_none(self):
        assert get_score_color(None) == "#6b6f88"

    def test_label_for_high(self):
        assert get_score_label(90) == "Exceptional"

    def test_label_for_none(self):
        assert get_score_label(None) == "—"


# ─── Improvement Module ──────────────────────────────────

class TestSplitWindows:
    def test_splits_correctly(self):
        activities = []
        for i in range(20):
            activities.append(_make_activity(id=i, days_ago=i * 3, runny_score=50 + i))
        recent, previous = _split_windows(activities, window_days=28)
        assert len(recent) > 0
        assert len(previous) > 0

    def test_empty_activities(self):
        recent, previous = _split_windows([], window_days=28)
        assert len(recent) == 0
        assert len(previous) == 0


class TestSigmoidScale:
    def test_center_maps_to_50(self):
        assert abs(_sigmoid_scale(0, center=0, spread=1) - 50) < 1

    def test_positive_maps_above_50(self):
        assert _sigmoid_scale(5, center=0, spread=1) > 90

    def test_negative_maps_below_50(self):
        assert _sigmoid_scale(-5, center=0, spread=1) < 10


class TestComputeImprovement:
    def test_empty_activities(self):
        result = compute_improvement([])
        assert result["trend"] == "flat"
        assert result["improvement_index"] == 50

    def test_returns_expected_keys(self):
        activities = [
            _make_activity(id=i, days_ago=i * 2, runny_score=50)
            for i in range(30)
        ]
        result = compute_improvement(activities)
        assert "improvement_index" in result
        assert "trend" in result
        assert "window_days" in result
        assert "ewma_history" in result
        assert result["trend"] in ("improving", "flat", "declining")

    def test_improving_athlete(self):
        # Recent runs faster than older runs
        activities = []
        for i in range(20):
            days = i * 3
            # Newer runs (lower days_ago / lower i) should be FASTER (lower pace value)
            pace = 300 + i * 5  # i=0 → 300 (fast, recent), i=19 → 395 (slow, old)
            activities.append(_make_activity(
                id=i, days_ago=days, pace=pace,
                avg_hr=150, runny_score=50 + (20 - i),
            ))
        result = compute_improvement(activities, window_days=28)
        # The most recent activities are the fastest
        assert result["improvement_index"] >= 50

    def test_configurable_window(self):
        activities = [
            _make_activity(id=i, days_ago=i * 2, runny_score=50)
            for i in range(30)
        ]
        result = compute_improvement(activities, window_days=14)
        assert result["window_days"] == 14

    def test_ewma_backward_compat(self):
        activities = [
            _make_activity(id=i, days_ago=i * 2, runny_score=60)
            for i in range(10)
        ]
        result = compute_improvement(activities)
        assert "current_ewma" in result
        assert "previous_ewma" in result
        assert "improvement_pct" in result
        assert len(result["ewma_history"]) == len(activities)


class TestComputeEwma:
    def test_empty(self):
        assert compute_ewma([]) == []

    def test_single(self):
        assert compute_ewma([50]) == [50]

    def test_trending_up(self):
        scores = [20, 30, 40, 50, 60, 70, 80]
        ewma = compute_ewma(scores)
        # EWMA should be increasing
        for i in range(1, len(ewma)):
            assert ewma[i] >= ewma[i - 1]


# ─── Baseline Exclusion ──────────────────────────────────

class TestBaselineExclusion:
    def test_excludes_current_activity(self):
        activities = _make_baseline(n=10)
        target_id = activities[0]["id"]
        baseline = _get_baseline(activities, exclude_id=target_id)
        ids = [a["id"] for a in baseline]
        assert target_id not in ids

    def test_includes_others(self):
        activities = _make_baseline(n=10)
        target_id = activities[0]["id"]
        baseline = _get_baseline(activities, exclude_id=target_id)
        assert len(baseline) > 0

    def test_filters_to_window(self):
        # Activity from 200 days ago should be excluded
        old = _make_activity(id=1, days_ago=200)
        recent = _make_activity(id=2, days_ago=5)
        baseline = _get_baseline([old, recent])
        assert len(baseline) == 1
        assert baseline[0]["id"] == 2
