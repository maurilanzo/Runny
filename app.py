"""
Runny – Strava-Powered Running Score & Improvement Tracker
Main Flask application
"""
import os
import json
import math
from flask import Flask, render_template, redirect, request, session, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()

from db import init_db, get_db
from strava_api import (
    get_auth_url,
    exchange_token,
    refresh_access_token,
    fetch_all_activities,
    fetch_activity_detail,
    fetch_activity_streams,
    fetch_activity_laps,
)
from scoring import score_activity, score_all_activities, get_score_color, get_score_label
from improvement import compute_improvement, get_trend_color, get_trend_icon, DEFAULT_TREND_WINDOW_DAYS

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# Initialize database on startup
with app.app_context():
    init_db()


# ─── Helpers ──────────────────────────────────────────────

def require_auth(f):
    """Decorator that ensures the user is authenticated."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if "access_token" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def format_pace(seconds_per_km):
    """Format pace as M:SS /km."""
    if not seconds_per_km or seconds_per_km <= 0:
        return "—"
    mins = int(seconds_per_km // 60)
    secs = int(seconds_per_km % 60)
    return f"{mins}:{secs:02d}"


def format_duration(seconds):
    """Format duration as HH:MM:SS or MM:SS."""
    if not seconds:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_distance(meters):
    """Format distance in km."""
    if not meters:
        return "—"
    return f"{meters / 1000:.2f}"


def format_speed(seconds_per_km):
    """Format pace (seconds per km) to speed in km/h."""
    if not seconds_per_km or seconds_per_km <= 0:
        return "—"
    return f"{3600 / seconds_per_km:.1f}"


# Register template filters
app.jinja_env.filters["pace"] = format_pace
app.jinja_env.filters["speed"] = format_speed
app.jinja_env.filters["duration"] = format_duration
app.jinja_env.filters["distance_km"] = format_distance
app.jinja_env.globals["get_score_color"] = get_score_color
app.jinja_env.globals["get_score_label"] = get_score_label
app.jinja_env.globals["get_trend_color"] = get_trend_color
app.jinja_env.globals["get_trend_icon"] = get_trend_icon


# ─── Auth Routes ──────────────────────────────────────────

@app.route("/")
def index():
    if "access_token" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/auth/strava")
def auth_strava():
    """Redirect to Strava OAuth."""
    return redirect(get_auth_url())


@app.route("/callback")
def callback():
    """Handle Strava OAuth callback."""
    code = request.args.get("code")
    if not code:
        return redirect(url_for("login"))

    try:
        tokens = exchange_token(code)
        session["access_token"] = tokens["access_token"]
        session["refresh_token"] = tokens["refresh_token"]
        session["expires_at"] = tokens["expires_at"]
        session["athlete"] = tokens.get("athlete", {})

        # Save athlete info to db
        db = get_db()
        athlete = tokens.get("athlete", {})
        db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            ("athlete_name", f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}"),
        )
        db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            ("athlete_avatar", athlete.get("profile", "")),
        )
        db.commit()

        return redirect(url_for("dashboard"))
    except Exception as e:
        return render_template("login.html", error=str(e))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─── Ensure valid token ──────────────────────────────────

def ensure_token():
    """Refresh the access token if expired."""
    import time

    expires_at = session.get("expires_at", 0)
    if time.time() >= expires_at - 60:
        try:
            tokens = refresh_access_token(session["refresh_token"])
            session["access_token"] = tokens["access_token"]
            session["refresh_token"] = tokens["refresh_token"]
            session["expires_at"] = tokens["expires_at"]
        except Exception:
            session.clear()
            return False
    return True


# ─── Dashboard ────────────────────────────────────────────

@app.route("/dashboard")
@require_auth
def dashboard():
    if not ensure_token():
        return redirect(url_for("login"))

    db = get_db()
    
    # Get current sport
    sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
    current_sport = sport_row["value"] if sport_row else "Run"
    sport_types = "('Ride', 'VirtualRide', 'MountainBikeRide', 'GravelRide', 'E-BikeRide')" if current_sport == "Ride" else "('Run', 'TrailRun')"

    activities = db.execute(
        f"SELECT * FROM activities WHERE sport_type IN {sport_types} ORDER BY start_date DESC"
    ).fetchall()
    activities = [dict(a) for a in activities]

    # Compute improvement with configurable window
    window_row = db.execute(
        "SELECT value FROM settings WHERE key = 'trend_window_days'"
    ).fetchone()
    window_days = int(window_row["value"]) if window_row else DEFAULT_TREND_WINDOW_DAYS
    improvement = compute_improvement(activities, window_days=window_days)

    # Stats
    total_runs = len(activities)
    scored = [a for a in activities if a.get("runny_score") is not None]
    avg_score = round(sum(a["runny_score"] for a in scored) / len(scored), 1) if scored else 0
    best_score = max((a["runny_score"] for a in scored), default=0)
    recent = activities[:5]

    # Athlete info
    athlete_name = db.execute(
        "SELECT value FROM settings WHERE key = 'athlete_name'"
    ).fetchone()
    athlete_avatar = db.execute(
        "SELECT value FROM settings WHERE key = 'athlete_avatar'"
    ).fetchone()

    return render_template(
        "dashboard.html",
        total_runs=total_runs,
        avg_score=avg_score,
        best_score=best_score,
        improvement=improvement,
        window_days=window_days,
        recent=recent,
        athlete_name=athlete_name["value"] if athlete_name else "",
        athlete_avatar=athlete_avatar["value"] if athlete_avatar else "",
        current_sport=current_sport,
    )


# ─── Activities ───────────────────────────────────────────

@app.route("/activities")
@require_auth
def activities():
    if not ensure_token():
        return redirect(url_for("login"))

    db = get_db()
    
    # Get current sport
    sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
    current_sport = sport_row["value"] if sport_row else "Run"
    sport_types = "('Ride', 'VirtualRide', 'MountainBikeRide', 'GravelRide', 'E-BikeRide')" if current_sport == "Ride" else "('Run', 'TrailRun')"

    rows = db.execute(
        f"SELECT * FROM activities WHERE sport_type IN {sport_types} ORDER BY start_date DESC"
    ).fetchall()
    activities_list = [dict(a) for a in rows]

    return render_template("activities.html", activities=activities_list, current_sport=current_sport)


@app.route("/activity/<int:activity_id>")
@require_auth
def activity_detail(activity_id):
    if not ensure_token():
        return redirect(url_for("login"))

    db = get_db()
    activity = db.execute(
        "SELECT * FROM activities WHERE id = ?", (activity_id,)
    ).fetchone()

    if not activity:
        return redirect(url_for("activities"))

    activity = dict(activity)

    # Recompute score breakdown
    all_acts = [dict(a) for a in db.execute(
        "SELECT * FROM activities ORDER BY start_date DESC"
    ).fetchall()]
    _, breakdown = score_activity(activity, all_acts, db=db)
    activity["breakdown"] = breakdown

    # Get current sport
    sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
    current_sport = sport_row["value"] if sport_row else "Run"

    return render_template("detail.html", activity=activity, current_sport=current_sport)


# ─── API Endpoints ────────────────────────────────────────

@app.route("/api/sync/preview", methods=["POST"])
@require_auth
def sync_preview():
    """Fetch activities from Strava and return the new ones for selection."""
    if not ensure_token():
        return jsonify({"error": "Authentication expired"}), 401

    try:
        db = get_db()
        sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
        current_sport = sport_row["value"] if sport_row else "Run"

        raw_activities = fetch_all_activities(session["access_token"], active_sport=current_sport)

        existing = set(
            r["id"]
            for r in db.execute("SELECT id FROM activities").fetchall()
        )

        new_activities = []
        for a in raw_activities:
            if a["id"] not in existing:
                distance = a.get("distance", 0)
                moving_time = a.get("moving_time", 0)
                pace = (moving_time / (distance / 1000)) if distance > 0 else 0
                new_activities.append({
                    "id": a["id"],
                    "name": a.get("name", "Run"),
                    "start_date": a.get("start_date", ""),
                    "distance": distance,
                    "moving_time": moving_time,
                    "pace": pace,
                    "total_elevation_gain": a.get("total_elevation_gain", 0),
                    "average_heartrate": a.get("average_heartrate"),
                    "sport_type": a.get("sport_type", a.get("type", "Run")),
                })

        return jsonify({"activities": new_activities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sync", methods=["POST"])
@require_auth
def sync_activities():
    """Import selected activities from the preview list."""
    if not ensure_token():
        return jsonify({"error": "Authentication expired"}), 401

    try:
        data = request.get_json()
        selected_ids = set(data.get("ids", []))

        if not selected_ids:
            return jsonify({"error": "No activities selected"}), 400

        db = get_db()
        sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
        current_sport = sport_row["value"] if sport_row else "Run"
        
        raw_activities = fetch_all_activities(session["access_token"], active_sport=current_sport)

        imported = 0
        for a in raw_activities:
            if a["id"] not in selected_ids:
                continue

            distance = a.get("distance", 0)
            moving_time = a.get("moving_time", 0)
            pace = (moving_time / (distance / 1000)) if distance > 0 else 0

            db.execute(
                """INSERT OR REPLACE INTO activities
                   (id, name, start_date, distance, moving_time, elapsed_time,
                    total_elevation_gain, average_heartrate, max_heartrate,
                    pace, sport_type, rpe, training_type, runny_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    a["id"],
                    a.get("name", "Run"),
                    a.get("start_date", ""),
                    distance,
                    moving_time,
                    a.get("elapsed_time", 0),
                    a.get("total_elevation_gain", 0),
                    a.get("average_heartrate"),
                    a.get("max_heartrate"),
                    pace,
                    a.get("sport_type", a.get("type", "Run")),
                    None,
                    None,
                    None,
                ),
            )
            imported += 1
        db.commit()

        # Score all activities corresponding to this sport type
        sport_types = "('Ride', 'VirtualRide', 'MountainBikeRide', 'GravelRide', 'E-BikeRide')" if current_sport == "Ride" else "('Run', 'TrailRun')"
        all_acts = [dict(r) for r in db.execute(
            f"SELECT * FROM activities WHERE sport_type IN {sport_types} ORDER BY start_date DESC"
        ).fetchall()]
        scored = score_all_activities(all_acts, db=db)

        for a in scored:
            db.execute(
                "UPDATE activities SET runny_score = ? WHERE id = ?",
                (a["runny_score"], a["id"]),
            )
        db.commit()

        return jsonify({"synced": imported, "total": len(all_acts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/activity/<int:activity_id>/delete", methods=["DELETE", "POST"])
@require_auth
def delete_activity(activity_id):
    """Delete a single activity."""
    db = get_db()
    db.execute("DELETE FROM activities WHERE id = ?", (activity_id,))
    db.commit()
    return jsonify({"deleted": activity_id})


@app.route("/api/activity/<int:activity_id>/update", methods=["POST"])
@require_auth
def update_activity(activity_id):
    """Update activity parameters and re-score."""
    data = request.get_json()

    db = get_db()
    activity = db.execute("SELECT * FROM activities WHERE id = ?", (activity_id,)).fetchone()
    if not activity:
        return jsonify({"error": "Activity not found"}), 404
        
    activity = dict(activity)

    new_rpe = data.get("rpe", activity["rpe"])
    new_training_type = data.get("training_type", activity["training_type"])
    
    new_name = data.get("name", activity["name"])
    new_distance = float(data.get("distance")) if data.get("distance") is not None else activity["distance"]
    new_moving_time = int(data.get("moving_time")) if data.get("moving_time") is not None else activity["moving_time"]
    new_elevation = float(data.get("total_elevation_gain")) if data.get("total_elevation_gain") is not None else activity["total_elevation_gain"]
    
    new_pace = (new_moving_time / (new_distance / 1000)) if new_distance and new_distance > 0 else 0
    
    is_modified = activity.get("is_modified", 0)
    if any(k in data for k in ("name", "distance", "moving_time", "total_elevation_gain")):
        is_modified = 1

    db.execute(
        """UPDATE activities 
           SET rpe = ?, training_type = ?, name = ?, distance = ?, 
               moving_time = ?, total_elevation_gain = ?, pace = ?, is_modified = ? 
           WHERE id = ?""",
        (new_rpe, new_training_type, new_name, new_distance, new_moving_time, 
         new_elevation, new_pace, is_modified, activity_id),
    )
    db.commit()

    # Re-score this activity
    activity = dict(
        db.execute("SELECT * FROM activities WHERE id = ?", (activity_id,)).fetchone()
    )
    all_acts = [dict(r) for r in db.execute(
        "SELECT * FROM activities ORDER BY start_date DESC"
    ).fetchall()]

    new_score, breakdown = score_activity(activity, all_acts, db=db)
    db.execute(
        "UPDATE activities SET runny_score = ? WHERE id = ?",
        (new_score, activity_id),
    )
    db.commit()

    return jsonify({"score": new_score, "breakdown": breakdown})


# ─── Streams Endpoint ─────────────────────────────────────

@app.route("/api/activity/<int:activity_id>/streams")
@require_auth
def activity_streams(activity_id):
    """Return stream data for the run analysis chart (cached)."""
    if not ensure_token():
        return jsonify({"error": "Authentication expired"}), 401

    db = get_db()

    # Check cache first
    cached = db.execute(
        "SELECT stream_data FROM streams WHERE activity_id = ?",
        (activity_id,),
    ).fetchone()

    if cached:
        return jsonify(json.loads(cached["stream_data"]))

    # Fetch from Strava
    try:
        streams = fetch_activity_streams(activity_id, session["access_token"])

        # Cache it
        db.execute(
            "INSERT OR REPLACE INTO streams (activity_id, stream_data) VALUES (?, ?)",
            (activity_id, json.dumps(streams)),
        )
        db.commit()

        return jsonify(streams)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/activity/<int:activity_id>/laps")
@require_auth
def activity_laps(activity_id):
    """Return laps data for the run analysis chart (cached)."""
    if not ensure_token():
        return jsonify({"error": "Authentication expired"}), 401

    db = get_db()

    # Check cache first
    cached = db.execute(
        "SELECT laps_data FROM laps WHERE activity_id = ?",
        (activity_id,),
    ).fetchone()

    if cached:
        return jsonify(json.loads(cached["laps_data"]))

    # Fetch from Strava
    try:
        laps = fetch_activity_laps(activity_id, session["access_token"])

        # Cache it
        db.execute(
            "INSERT OR REPLACE INTO laps (activity_id, laps_data) VALUES (?, ?)",
            (activity_id, json.dumps(laps)),
        )
        db.commit()

        return jsonify(laps)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Settings Endpoint ────────────────────────────────────

@app.route("/api/settings/trend-window", methods=["POST"])
@require_auth
def update_trend_window():
    """Update the trend comparison window size."""
    data = request.get_json()
    days = int(data.get("days", DEFAULT_TREND_WINDOW_DAYS))
    days = max(7, min(90, days))  # Clamp to 7–90 range
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        ("trend_window_days", str(days)),
    )
    db.commit()
    return jsonify({"trend_window_days": days})

@app.route("/api/settings/sport", methods=["POST"])
@require_auth
def update_current_sport():
    """Toggle current sport between Run and Ride."""
    data = request.get_json()
    sport = data.get("sport", "Run")
    if sport not in ("Run", "Ride"):
        sport = "Run"
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        ("current_sport", sport),
    )
    db.commit()
    return jsonify({"current_sport": sport})


# ─── Chart Data Endpoint ─────────────────────────────────

@app.route("/api/chart-data")
@require_auth
def chart_data():
    db = get_db()
    sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
    current_sport = sport_row["value"] if sport_row else "Run"
    sport_types = "('Ride', 'VirtualRide', 'MountainBikeRide', 'GravelRide', 'E-BikeRide')" if current_sport == "Ride" else "('Run', 'TrailRun')"

    activities = [dict(a) for a in db.execute(
        f"SELECT * FROM activities WHERE sport_type IN {sport_types} ORDER BY start_date ASC"
    ).fetchall()]

    # Read configurable window
    window_row = db.execute(
        "SELECT value FROM settings WHERE key = 'trend_window_days'"
    ).fetchone()
    window_days = int(window_row["value"]) if window_row else DEFAULT_TREND_WINDOW_DAYS

    improvement = compute_improvement(activities, window_days=window_days)
    return jsonify(improvement)


# ─── Comparison ──────────────────────────────────────────

@app.route("/compare")
@require_auth
def compare_view():
    if not ensure_token():
        return redirect(url_for("login"))

    db = get_db()
    sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
    current_sport = sport_row["value"] if sport_row else "Run"
    sport_types = "('Ride', 'VirtualRide', 'MountainBikeRide', 'GravelRide', 'E-BikeRide')" if current_sport == "Ride" else "('Run', 'TrailRun')"

    rows = db.execute(
        f"SELECT * FROM activities WHERE sport_type IN {sport_types} ORDER BY start_date DESC"
    ).fetchall()
    activities_list = [dict(a) for a in rows]

    return render_template("compare.html", activities=activities_list, current_sport=current_sport)


@app.route("/api/compare/<int:activity_id>/suggestions")
@require_auth
def compare_suggestions(activity_id):
    """Return activities of the same training type, ranked by similarity.

    Similarity is the Euclidean distance of min-max-normalized
    (avg_heartrate, distance, pace) vectors.
    """
    db = get_db()
    sport_row = db.execute("SELECT value FROM settings WHERE key = 'current_sport'").fetchone()
    current_sport = sport_row["value"] if sport_row else "Run"
    sport_types = "('Ride', 'VirtualRide', 'MountainBikeRide', 'GravelRide', 'E-BikeRide')" if current_sport == "Ride" else "('Run', 'TrailRun')"

    target = db.execute("SELECT * FROM activities WHERE id = ?", (activity_id,)).fetchone()
    if not target:
        return jsonify({"error": "Activity not found"}), 404
    target = dict(target)

    training_type = target.get("training_type")
    if not training_type:
        return jsonify({"error": "Source activity has no training type"}), 400

    # Fetch all activities of the same type (excluding the target)
    rows = db.execute(
        f"""SELECT * FROM activities
            WHERE sport_type IN {sport_types}
              AND training_type = ?
              AND id != ?
            ORDER BY start_date DESC""",
        (training_type, activity_id),
    ).fetchall()
    candidates = [dict(r) for r in rows]

    if not candidates:
        return jsonify({"suggestions": [], "target": target})

    # Compute pairwise percentage-difference similarity.
    # For each metric, diff = |a - b| / max(a, b), giving 0-1.
    # Similarity = (1 - avg_diff) * 100.
    t_hr = target.get("average_heartrate") or 0
    t_dist = target.get("distance") or 0
    t_pace = target.get("pace") or 0

    def pct_diff(a, b):
        """Return 0-1 fractional difference between two values."""
        if a == 0 and b == 0:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b))

    scored = []
    for cand in candidates:
        c_hr = cand.get("average_heartrate") or 0
        c_dist = cand.get("distance") or 0
        c_pace = cand.get("pace") or 0

        diffs = []
        if t_hr > 0 or c_hr > 0:
            diffs.append(pct_diff(t_hr, c_hr))
        if t_dist > 0 or c_dist > 0:
            diffs.append(pct_diff(t_dist, c_dist))
        if t_pace > 0 or c_pace > 0:
            diffs.append(pct_diff(t_pace, c_pace))

        avg_diff = sum(diffs) / len(diffs) if diffs else 1.0
        similarity_pct = round((1 - avg_diff) * 100, 1)

        c = dict(cand)
        c["similarity_pct"] = max(0, similarity_pct)
        scored.append(c)

    scored.sort(key=lambda x: -x["similarity_pct"])  # highest similarity first

    return jsonify({"suggestions": scored[:20], "target": target})


# ─── Run ──────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5050)
