"""
Strava API client – OAuth2 and activity fetching.
"""
import os
import requests

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


def fetch_all_activities(access_token: str) -> list:
    """
    Fetch ALL running activities (paginated).
    Filters to Run and TrailRun sport types only.
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

        # Filter to runs only
        runs = [
            a
            for a in batch
            if a.get("type") == "Run"
            or a.get("sport_type") in ("Run", "TrailRun")
        ]
        all_activities.extend(runs)

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
