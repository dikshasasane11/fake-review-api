"""
Database module — SQLite analytics storage.
Tracks every prediction made via the API.
"""

import sqlite3
import json
import os
from datetime import datetime, date, timedelta
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.db")


class Database:

    def init(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  TEXT    NOT NULL,
                    date        TEXT    NOT NULL,
                    hour        INTEGER NOT NULL,
                    text_len    INTEGER NOT NULL,
                    rating      INTEGER NOT NULL,
                    is_fake     INTEGER NOT NULL,
                    confidence  REAL    NOT NULL,
                    latency_ms  REAL    NOT NULL,
                    word_count  INTEGER NOT NULL,
                    exclamations INTEGER NOT NULL,
                    unique_ratio REAL   NOT NULL,
                    superlatives INTEGER NOT NULL
                )
            """)
        print("Database initialised at", DB_PATH)

    def _conn(self):
        return sqlite3.connect(DB_PATH)

    def log_prediction(self, text, rating, is_fake, confidence, latency_ms, signals):
        now = datetime.utcnow()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO predictions
                (created_at, date, hour, text_len, rating, is_fake, confidence,
                 latency_ms, word_count, exclamations, unique_ratio, superlatives)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                now.hour,
                len(text),
                rating,
                int(is_fake),
                confidence,
                latency_ms,
                signals.get("word_count", 0),
                signals.get("exclamation_count", 0),
                signals.get("unique_word_ratio", 0),
                signals.get("superlative_count", 0),
            ))

    def get_analytics(self):
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Overall totals
            cur.execute("SELECT COUNT(*) as total, SUM(is_fake) as fakes, AVG(confidence) as avg_conf, AVG(latency_ms) as avg_lat FROM predictions")
            row = cur.fetchone()
            total      = row["total"] or 0
            fakes      = int(row["fakes"] or 0)
            genuine    = total - fakes
            avg_conf   = round((row["avg_conf"] or 0) * 100, 1)
            avg_lat    = round(row["avg_lat"] or 0, 1)

            # Last 7 days daily counts
            days = [(date.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
            daily = {}
            cur.execute("""
                SELECT date, SUM(is_fake) as fakes, COUNT(*) as total
                FROM predictions
                WHERE date >= ?
                GROUP BY date
            """, (days[0],))
            for r in cur.fetchall():
                daily[r["date"]] = {"fake": int(r["fakes"]), "total": int(r["total"])}

            daily_data = [
                {
                    "date": d,
                    "label": datetime.strptime(d, "%Y-%m-%d").strftime("%b %d"),
                    "total": daily.get(d, {}).get("total", 0),
                    "fake":  daily.get(d, {}).get("fake",  0),
                    "genuine": daily.get(d, {}).get("total", 0) - daily.get(d, {}).get("fake", 0),
                }
                for d in days
            ]

            # Hourly distribution (last 24h)
            cur.execute("""
                SELECT hour, COUNT(*) as cnt FROM predictions
                WHERE created_at >= datetime("now", "-1 day")
                GROUP BY hour ORDER BY hour
            """)
            hourly = {r["hour"]: r["cnt"] for r in cur.fetchall()}
            hourly_data = [{"hour": h, "count": hourly.get(h, 0)} for h in range(24)]

            # Rating distribution
            cur.execute("SELECT rating, COUNT(*) as cnt, SUM(is_fake) as fakes FROM predictions GROUP BY rating ORDER BY rating")
            rating_data = [{"rating": r["rating"], "total": r["cnt"], "fake": int(r["fakes"] or 0)} for r in cur.fetchall()]

            # Recent 10 predictions
            cur.execute("""
                SELECT created_at, is_fake, confidence, word_count, exclamations, rating,
                       SUBSTR(created_at, 1, 19) as ts
                FROM predictions ORDER BY id DESC LIMIT 10
            """)
            recent = [
                {
                    "time":       r["ts"].replace("T", " "),
                    "is_fake":    bool(r["is_fake"]),
                    "label":      "FAKE" if r["is_fake"] else "GENUINE",
                    "confidence": round(r["confidence"] * 100, 1),
                    "word_count": r["word_count"],
                    "rating":     r["rating"],
                }
                for r in cur.fetchall()
            ]

            # Signal flags summary
            cur.execute("SELECT SUM(CASE WHEN word_count < 20 THEN 1 ELSE 0 END) as short_reviews, SUM(CASE WHEN exclamations > 3 THEN 1 ELSE 0 END) as excess_excl, SUM(CASE WHEN unique_ratio < 0.6 THEN 1 ELSE 0 END) as repetitive, SUM(CASE WHEN superlatives > 2 THEN 1 ELSE 0 END) as excess_sup FROM predictions")
            sig = cur.fetchone()
            signals_summary = {
                "short_reviews":    int(sig["short_reviews"] or 0),
                "excess_exclamations": int(sig["excess_excl"] or 0),
                "repetitive_language": int(sig["repetitive"] or 0),
                "excess_superlatives": int(sig["excess_sup"] or 0),
            }

        return {
            "total":           total,
            "fake_count":      fakes,
            "genuine_count":   genuine,
            "fake_pct":        round(fakes / total * 100, 1) if total else 0,
            "genuine_pct":     round(genuine / total * 100, 1) if total else 0,
            "avg_confidence":  avg_conf,
            "avg_latency_ms":  avg_lat,
            "daily":           daily_data,
            "hourly":          hourly_data,
            "rating_dist":     rating_data,
            "recent":          recent,
            "signals_summary": signals_summary,
        }
