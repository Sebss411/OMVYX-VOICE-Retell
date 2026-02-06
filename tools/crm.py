"""
Omvyx Voice — CRM Tool (Mock)

Simulates a CRM backend for user lookup and creation.
Replace the mock dictionaries with real DB/API calls in production.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Mock database — keyed by DNI
# ---------------------------------------------------------------------------

_MOCK_USERS: dict[str, dict] = {
    "12345678A": {
        "name": "María García",
        "dni": "12345678A",
        "email": "maria@example.com",
        "phone": "+34600111222",
    },
    "87654321B": {
        "name": "Carlos López",
        "dni": "87654321B",
        "email": "carlos@example.com",
        "phone": "+34600333444",
    },
}


async def lookup_user(dni: str) -> dict | None:
    """Search for an existing user by DNI.  Returns the user dict or None."""
    return _MOCK_USERS.get(dni.strip().upper())


async def create_user(
    name: str, dni: str, email: str, phone: str
) -> dict:
    """Register a new user and return the created record."""
    record = {
        "name": name.strip(),
        "dni": dni.strip().upper(),
        "email": email.strip().lower(),
        "phone": phone.strip(),
    }
    _MOCK_USERS[record["dni"]] = record
    return record
