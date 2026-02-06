"""
Omvyx Voice — Calendar Tool (Mock)

Simulates appointment availability checks.
Smart logic: when the requested slot is busy, the tool returns the next
2 available alternatives so the LLM can offer them immediately
without an extra round-trip.
"""

from __future__ import annotations

from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Mock busy slots
# ---------------------------------------------------------------------------

_BUSY_SLOTS: set[str] = {
    "2026-02-09 10:00",
    "2026-02-09 11:00",
    "2026-02-10 09:00",
}

# Business hours: 09:00 – 17:00, 1-hour slots, Mon–Fri
_OPEN_HOUR = 9
_CLOSE_HOUR = 17


def _is_business_slot(dt: datetime) -> bool:
    return dt.weekday() < 5 and _OPEN_HOUR <= dt.hour < _CLOSE_HOUR


def _next_available(after: datetime, count: int = 2) -> list[str]:
    """Return the next `count` available business-hour slots after `after`."""
    slots: list[str] = []
    candidate = after.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while len(slots) < count:
        if _is_business_slot(candidate):
            key = candidate.strftime("%Y-%m-%d %H:%M")
            if key not in _BUSY_SLOTS:
                slots.append(key)
        candidate += timedelta(hours=1)
        # safety: don't loop forever
        if candidate > after + timedelta(days=30):
            break
    return slots


async def check_availability(date_str: str) -> dict:
    """
    Check if `date_str` (format "YYYY-MM-DD HH:MM") is available.

    Returns:
        {
            "available": bool,
            "requested": str,
            "alternatives": list[str]   # only present when unavailable
        }
    """
    key = date_str.strip()
    try:
        dt = datetime.strptime(key, "%Y-%m-%d %H:%M")
    except ValueError:
        return {
            "available": False,
            "requested": key,
            "error": "Invalid date format. Use YYYY-MM-DD HH:MM.",
        }

    if not _is_business_slot(dt):
        return {
            "available": False,
            "requested": key,
            "error": "Outside business hours (Mon-Fri, 09:00-17:00).",
            "alternatives": _next_available(dt),
        }

    if key in _BUSY_SLOTS:
        return {
            "available": False,
            "requested": key,
            "alternatives": _next_available(dt),
        }

    return {"available": True, "requested": key}


async def book_slot(date_str: str, user_dni: str) -> dict:
    """Confirm a booking.  In production this writes to the calendar backend."""
    _BUSY_SLOTS.add(date_str.strip())
    return {
        "booked": True,
        "slot": date_str.strip(),
        "user_dni": user_dni,
    }
