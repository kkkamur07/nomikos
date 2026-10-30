"""The async bcrypt wrappers agree with the sync versions they replace."""

from __future__ import annotations

from backend.users.application.password import (
    hash_password_async,
    verify_password,
    verify_password_async,
)


async def test_async_wrappers_hash_and_verify_off_the_event_loop() -> None:
    hashed = await hash_password_async("correct-horse-battery-staple")

    assert hashed.startswith("$2b$12$")
    assert await verify_password_async("correct-horse-battery-staple", hashed)
    assert not await verify_password_async("wrong password", hashed)
    assert verify_password("correct-horse-battery-staple", hashed)
