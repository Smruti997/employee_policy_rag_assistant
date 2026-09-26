"""Generate mock JWTs for the three test users.

Usage:
    python3 mint_tokens.py

Requires:
    pip install pyjwt python-dotenv

Reads JWT_SECRET from .env. Tokens are valid for 7 days.
"""

import os
import time
from pathlib import Path

try:
    import jwt
except ImportError:
    raise SystemExit("pyjwt not installed. Run: pip install pyjwt python-dotenv")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can come from the shell

SECRET = os.getenv("JWT_SECRET", "change-me-in-env")
ALGO = "HS256"
TTL_SECONDS = 7 * 24 * 60 * 60

USERS = [
    {"sub": "emp-001",  "email": "emp@test.com",  "department": "hr",   "level": 1},
    {"sub": "mgr-002",  "email": "mgr@test.com",  "department": "hr",   "level": 2},
    {"sub": "exec-003", "email": "exec@test.com", "department": "exec", "level": 3},
]


def main() -> None:
    now = int(time.time())
    print(f"JWT_SECRET: {SECRET}")
    print(f"Algorithm:  {ALGO}")
    print()
    for u in USERS:
        payload = {**u, "iat": now, "exp": now + TTL_SECONDS}
        token = jwt.encode(payload, SECRET, algorithm=ALGO)
        print(f"# {u['email']}  (dept={u['department']}, level={u['level']})")
        print(token)
        print()


if __name__ == "__main__":
    main()
