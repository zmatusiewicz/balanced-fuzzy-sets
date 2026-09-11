"""Guarantee source labels used by fuzzy operator objects."""

from __future__ import annotations

BY_CONSTRUCTION = "by construction"
SYMBOLICALLY_VERIFIED = "symbolically verified"
NUMERICALLY_SAMPLED = "numerically sampled"
USER_SUPPLIED_UNCHECKED = "user supplied, unchecked"

VALID_GUARANTEES = frozenset(
    {
        BY_CONSTRUCTION,
        SYMBOLICALLY_VERIFIED,
        NUMERICALLY_SAMPLED,
        USER_SUPPLIED_UNCHECKED,
    }
)


def normalize_guarantee(guarantee: str | None, *, validate: bool) -> str:
    """Return a guarantee label compatible with the validation mode."""

    if guarantee is None:
        return NUMERICALLY_SAMPLED if validate else USER_SUPPLIED_UNCHECKED
    if guarantee not in VALID_GUARANTEES:
        raise ValueError(
            f"guarantee must be one of {sorted(VALID_GUARANTEES)!r}, got {guarantee!r}."
        )
    return guarantee


def attach_guarantee(function, guarantee: str, *, validate: bool = True):
    """Attach a guarantee label to a returned callable operator."""

    function.guarantee = normalize_guarantee(guarantee, validate=validate)
    return function
