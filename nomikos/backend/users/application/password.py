"""Password hashing via the bcrypt algorithm (not encryption).

We use the ``bcrypt`` library directly (bcrypt 4.x; passlib is not used).

- **Algorithm**: bcrypt (adaptive one-way password hash)
- **Salt / cost**: ``bcrypt.gensalt()`` → default **12 rounds** (2^12 iterations)
- **Stored form**: ASCII string like ``$2b$12$...`` (version, cost, salt, hash)
- **Verification**: ``bcrypt.checkpw`` (constant-time compare of hash)

Passwords are never stored in plaintext; only the bcrypt hash is persisted in ``users.hashed_password``.
"""

import asyncio

import bcrypt

# bcrypt default work factor when using gensalt() without rounds=
BCRYPT_ROUNDS = 12
MAX_BCRYPT_PASSWORD_BYTES = 72


def password_bytes(plain: str) -> bytes:
    encoded = plain.encode("utf-8")
    if len(encoded) > MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError("Password exceeds the bcrypt UTF-8 byte limit")
    return encoded


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(password_bytes(plain), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        encoded = password_bytes(plain)
    except ValueError:
        return False
    return bcrypt.checkpw(encoded, hashed.encode("utf-8"))


async def hash_password_async(plain: str) -> str:
    """Hash off the event loop; bcrypt costs ~300ms at 12 rounds."""
    return await asyncio.to_thread(hash_password, plain)


async def verify_password_async(plain: str, hashed: str) -> bool:
    """Verify off the event loop; see :func:`hash_password_async`."""
    return await asyncio.to_thread(verify_password, plain, hashed)
