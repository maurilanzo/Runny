"""
Strava API client – OAuth2 and activity fetching.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"
REDIRECT_URI = "http://localhost:5050/callback"


def get_auth_url():
    """Build the Strava OAuth authorization URL."""
    params = {
        "client_id": STRAVA_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "activity:read_all",
        "approval_prompt": "auto",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{STRAVA_AUTH_URL}?{qs}"


def exchange_token(code: str) -> dict:
    """Exchange an authorization code for access + refresh tokens."""
    resp = requests.post(
        STRAVA_TOKEN_URL,
        json={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Refresh an expired access token."""
    resp = requests.post(
        STRAVA_TOKEN_URL,
        json={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _strava_get(endpoint: str, access_token: str, params: dict = None):
    """Make an authenticated GET request to the Strava API."""
    resp = requests.get(
        f"{STRAVA_API_BASE}{endpoint}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
        timeout=30,
    )
    if resp.status_code == 429:
        raise Exception("Rate limited by Strava. Please try again in a few minutes.")
    resp.raise_for_status()
    return resp.json()


def fetch_all_activities(access_token: str, active_sport: str = "Run") -> list:
    """
    Fetch ALL running or biking activities (paginated).
    Filters depending on `active_sport`.
    """
    all_activities = []
    page = 1
    per_page = 100

    while True:
        batch = _strava_get(
            "/athlete/activities",
            access_token,
            params={"page": page, "per_page": per_page},
        )
        if not batch:
            break

        # Filter to specific sport types
        filtered = []
        for a in batch:
            sport = a.get("sport_type") or a.get("type", "")
            if active_sport == "Run" and sport in ("Run", "TrailRun"):
                filtered.append(a)
            elif active_sport == "Ride" and sport in ("Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "E-BikeRide"):
                filtered.append(a)
                
        all_activities.extend(filtered)

        if len(batch) < per_page:
            break
        page += 1

    return all_activities


def fetch_activity_detail(activity_id: int, access_token: str) -> dict:
    """Fetch detailed info for a single activity."""
    return _strava_get(
        f"/activities/{activity_id}",
        access_token,
        params={"include_all_efforts": "false"},
    )


def fetch_activity_streams(activity_id: int, access_token: str) -> list:
    """
    Fetch time-series stream data for a single activity.

    Returns a list of stream dicts, each with:
        {"type": "velocity_smooth", "data": [...], "series_type": "distance", "resolution": "high"}

    Requested stream keys: time, distance, altitude, velocity_smooth,
    heartrate, cadence, grade_smooth.
    Not all keys may be present in the response (e.g. no heartrate
    if the athlete didn't wear an HR monitor).
    """
    keys = "time,distance,altitude,velocity_smooth,heartrate,cadence,grade_smooth"
    return _strava_get(
        f"/activities/{activity_id}/streams",
        access_token,
        params={"keys": keys, "key_by_type": "true"},
    )


def fetch_activity_laps(activity_id: int, access_token: str) -> list:
    """Fetch laps for a single activity."""
    return _strava_get(
        f"/activities/{activity_id}/laps",
        access_token,
    )
