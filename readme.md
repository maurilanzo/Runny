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

### Scoring Engine (`scoring.py`)
Provides a **Relative Session-Quality Score (0-100)** answering: *"How strong/efficient was this session relative to my recent comparable baseline?"*

| Factor | Aerobic Weight | Quality Weight | Description |
|---|---|---|---|
| Pace | 40% | 50% | Percentile rank against past comparable baseline. |
| Aerobic Efficiency | 30% | 20% | Speed/HR ratio, indicating cardiovascular efficiency. |
| Execution Quality | 20% | 20% | Variance penalizing (evaluates plausible intent-alignment). |
| Consistency | 10% | 10% | Variance of pace evaluating session stability (CV). |

Scores are **percentile-based** against the last 90 days. Falls back to global sport medians if the baseline is too sparse. Evaluates `aerobic` runs (easy, moderate, long) differently than `quality` sessions (tempo, intervals, race).

### Improvement Index (`improvement.py`)
Answers: *"Am I improving over time, and by how much?"*

Instead of simply averaging activity scores, it focuses on capability and aerobic development across two consecutive 42-day time windows.
- **Primary Signals**: Critical Speed (recent best efforts) & Aerobic Efficiency (submaximal runs).
- **Secondary Signals**: Adjusted aerobic pace.
- **Trend Logic**: 📈 Improving (Index ≥ 57), ➡️ Flat, 📉 Declining (Index ≤ 43).
- **Fallback**: EWMA is retained for charts and as a last-resort fallback.

### Activity Analysis (`detail.html`)
The frontend computes a suite of interval and effort metrics, notably calculating **Cardiac Drift (Aerobic Decoupling)** by evaluating the change in the HR/Speed ratio between the first and last quarters of an activity.

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
