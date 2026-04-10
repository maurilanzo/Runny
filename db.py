"""
Database initialization and connection management for Runny.
Uses SQLite for lightweight local persistence.
"""
import sqlite3
import os
from flask import g

DATABASE = os.path.join(os.path.dirname(__file__), "runny.db")


def get_db():
    """Get a database connection for the current request."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


def close_db(e=None):
    """Close the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create database tables if they don't exist."""
    from flask import current_app

    current_app.teardown_appcontext(close_db)

    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS activities (
            id              INTEGER PRIMARY KEY,
            name            TEXT NOT NULL,
            start_date      TEXT NOT NULL,
            distance        REAL DEFAULT 0,
            moving_time     INTEGER DEFAULT 0,
            elapsed_time    INTEGER DEFAULT 0,
            total_elevation_gain REAL DEFAULT 0,
            average_heartrate REAL,
            max_heartrate    REAL,
            pace            REAL DEFAULT 0,
            sport_type      TEXT DEFAULT 'Run',
            rpe             INTEGER,
            training_type   TEXT,
            runny_score     INTEGER,
            imported_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS streams (
            activity_id  INTEGER PRIMARY KEY,
            stream_data  TEXT NOT NULL,
            fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS laps (
            activity_id  INTEGER PRIMARY KEY,
            laps_data    TEXT NOT NULL,
            fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_activities_date
            ON activities(start_date);
        CREATE INDEX IF NOT EXISTS idx_activities_score
            ON activities(runny_score);
    """
    )
    db.commit()
