# Runny – Implementation Walkthrough

## What Was Built

A **Python/Flask** web application that integrates with the **Strava API** to import running activities, score them with a weighted percentile system, and track athlete improvement over time.

## Project Structure

```
Runny/
├── app.py              ← Flask app with all routes
├── db.py               ← SQLite database layer
├── strava_api.py       ← Strava OAuth2 + API client
├── scoring.py          ← Weighted percentile scoring engine
├── improvement.py      ← EWMA-based improvement index
├── requirements.txt    ← Python dependencies
├── .env.example        ← Strava credential template
├── static/
│   ├── style.css       ← Dark-themed design system
│   └── favicon.svg     ← Brand icon
└── templates/
    ├── base.html       ← Shared layout
    ├── login.html      ← Strava connect page
    ├── dashboard.html  ← Stats + Chart.js trend chart
    ├── activities.html ← Activity list with scores
    └── detail.html     ← Activity detail + RPE editor
```

## Key Components

### Scoring Engine ([scoring.py](file:///Users/maurii/Desktop/Runny/scoring.py))
| Factor | Weight | Better = |
|---|---|---|
| Distance | 20% | Longer |
| Pace | 20% | Faster (inverted) |
| Elevation | 15% | More climbing |
| Duration | 10% | Longer |
| Avg HR | 10% | Higher effort |
| RPE | 10% | Higher perceived effort |
| Max HR | 5% | Higher peak |
| Training Type | 10% | Multiplier (Easy→0.8, Race→1.3) |

Scores are **percentile-based** against the last 90 days. Falls back to absolute benchmarks when < 5 activities exist.

### Improvement Index ([improvement.py](file:///Users/maurii/Desktop/Runny/improvement.py))
- **EWMA** (α=0.1) over all scored activities
- Compares current vs. 30-activities-ago EWMA
- Trend: 📈 improving (>+2%), ➡️ flat (±2%), 📉 declining (<-2%)

### Strava Integration ([strava_api.py](file:///Users/maurii/Desktop/Runny/strava_api.py))
- Full OAuth2 flow with auto token refresh
- Paginated activity fetch (filters to Run + TrailRun)
- Rate-limit handling


## How to Run

```bash
# 1. Copy and fill in your Strava credentials
cp .env.example .env
# Edit .env with your STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Start the app
python3 app.py

# 4. Open http://localhost:5050 in your browser
```

> [!IMPORTANT]
> You need a Strava API Application. Create one at [strava.com/settings/api](https://www.strava.com/settings/api) with callback domain set to `localhost`.
