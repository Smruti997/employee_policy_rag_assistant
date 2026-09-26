"""Generate mock JWTs for the mock login roster.

Usage:
    python3 mint_tokens.py

Requires:
    pip install -r requirements.txt

Reads JWT_SECRET from .env. Tokens are valid for 7 days.
"""

import os
import time

try:
    import jwt
except ImportError:
    raise SystemExit("pyjwt not installed. Run: pip install -r requirements.txt")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can come from the shell

SECRET = os.getenv("JWT_SECRET", "change-me-in-env")
ALGO = "HS256"
TTL_SECONDS = 7 * 24 * 60 * 60

# Every `sub` here must exist in the app/sample_data tables, because
# get_employee_context looks the user up by this id.
USERS = [
    {"sub": "emp-001", "email": "emp@test.com", "department": "hr", "level": 1},
    {"sub": "emp-002", "email": "mgr@test.com", "department": "hr", "level": 2},
    {"sub": "emp-003", "email": "exec@test.com", "department": "exec", "level": 3},
    {"sub": "emp-004", "email": "finance@test.com", "department": "finance", "level": 1}
]


def main() -> None:
    now = int(time.time())
    print(f"JWT_SECRET: {SECRET}")
    print(f"Algorithm:  {ALGO}")
    print()
    for user in USERS:
        payload = {**user, "iat": now, "exp": now + TTL_SECONDS}
        token = jwt.encode(payload, SECRET, algorithm=ALGO)
        print(f"# {user['sub']}  ({user['email']}, dept={user['department']}, level={user['level']})")
        print(token)
        print()


if __name__ == "__main__":
    main()
