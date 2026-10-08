"""Slots for one original plus five augmented copies, configurable by tier.

Every training line is shown once as the original (variant 0) and five times
as an augmented copy (variants 1..5). Each copy belongs to a tier, and the
number of operations per copy is fixed by its tier:

* ``none``: identical copy of the original, no augmentation at all
* ``easy``: three light ops
* ``mild``: two mild ops
* ``hard``: one heavy op

How many of the five copies go to each tier is set by a ``copies`` mapping
(for example from the augmentation config) and turned into a plan with
:func:`build_variant_plan`. The counts must sum to :data:`TOTAL_COPIES`.
"""

from __future__ import annotations

from collections.abc import Mapping

# Fixed number of operations applied to one copy of each tier, in plan order.
OPERATIONS_PER_STRENGTH: dict[str, int] = {"none": 0, "easy": 3, "mild": 2, "hard": 1}

# Number of augmented copies per line (the original is shown in addition).
TOTAL_COPIES = 5

# Default split of the five copies: 2 easy, 2 mild, 1 hard.
DEFAULT_COPIES: dict[str, int] = {"none": 0, "easy": 2, "mild": 2, "hard": 1}


def build_variant_plan(copies: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    """Turn per-tier copy counts into an ordered ``(strength, operation_count)`` plan.

    Missing tiers count as zero. The counts must be non-negative integers that
    sum to :data:`TOTAL_COPIES`. The plan is ordered none, easy, mild, hard.
    """
    unknown = sorted(set(copies) - set(OPERATIONS_PER_STRENGTH))
    if unknown:
        raise ValueError(
            f"Unknown augmentation tiers {unknown}; expected a subset of "
            f"{list(OPERATIONS_PER_STRENGTH)}."
        )
    for strength, count in copies.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Augmentation copies for {strength!r} must be a non-negative integer, "
                f"got {count!r}."
            )
    total = sum(copies.values())
    if total != TOTAL_COPIES:
        raise ValueError(
            f"Augmentation copies sum to {total}, but must sum to exactly {TOTAL_COPIES} "
            f"(got {dict(copies)})."
        )
    plan: list[tuple[str, int]] = []
    for strength, operation_count in OPERATIONS_PER_STRENGTH.items():
        plan.extend([(strength, operation_count)] * copies.get(strength, 0))
    return tuple(plan)


AUGMENTED_VARIANT_PLAN: tuple[tuple[str, int], ...] = build_variant_plan(DEFAULT_COPIES)

EXPECTED_N_AUGMENTATIONS = TOTAL_COPIES


def plan_for_augmented_variant(
    variant: int,
    plan: tuple[tuple[str, int], ...] = AUGMENTED_VARIANT_PLAN,
) -> tuple[str, int]:
    """Return ``(strength, operation_count)`` for an augmented variant index."""
    if variant < 1 or variant > len(plan):
        raise ValueError(f"Augmented variant must be between 1 and {len(plan)}, got {variant}.")
    return plan[variant - 1]
