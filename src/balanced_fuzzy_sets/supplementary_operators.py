"""Supplementary operator for balanced fuzzy sets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from .fuzzy_negations import (
    FuzzyNegation,
    complement_power,
    complement_root,
    cosine,
    power,
    root_complement,
    standard,
    sugeno,
    yager,
)
from .operator_guarantees import BY_CONSTRUCTION, normalize_guarantee

NumberFunction = Callable[[float], float]
EPS = 1e-9


@dataclass(frozen=True)
class SupplementaryOperator:
    """Concrete supplementary operator I:[-1,1]->[-1,1]."""

    name: str
    formula: str
    i: NumberFunction
    reference_note: str = ""
    guarantee: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=True),
        )

    def __call__(self, x: float) -> float:
        return _clamp_balanced(self.i(_balanced_interval(x)), f"{self.name}({x})")


def _balanced_interval(x: float, name: str = "x") -> float:
    if not -1.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1], got {x!r}")
    return x


def _clamp_balanced(x: float, name: str = "value") -> float:
    if not isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}.")
    if x < -1.0 - EPS or x > 1.0 + EPS:
        raise ValueError(f"{name} must be in [-1, 1], got {x!r}.")
    if x <= -1.0:
        return -1.0
    if x >= 1.0:
        return 1.0
    return x


def supplementary_operator(
    x: float,
    negation: NumberFunction | FuzzyNegation | None = None,
) -> float:
    """
    Evaluate the supplementary operator I from the attached paper.

    I(x)=n(x) for x>0, I(x)=-n(-x) for x<0, and I(0)=0.
    """

    if negation is None:
        negation = standard()

    x = _balanced_interval(x)
    n = negation if callable(negation) else negation.n

    if x > 0.0:
        return _clamp_balanced(n(x), "I(x)")
    if x < 0.0:
        return _clamp_balanced(-n(-x), "I(x)")
    return 0.0


def from_fuzzy_negation(
    negation: NumberFunction | FuzzyNegation,
    *,
    name: str | None = None,
    formula: str | None = None,
    reference_note: str = "",
) -> SupplementaryOperator:
    """Create supplementary operator I from a fuzzy negation n on [0,1]."""

    source_name = getattr(negation, "name", "fuzzy negation")
    source_formula = getattr(negation, "formula", "n(x)")

    return SupplementaryOperator(
        name=name or f"{source_name} supplementary operator",
        formula=formula or f"I(x)={source_formula} for x>0; mirrored for x<0",
        i=lambda x: supplementary_operator(x, negation),
        reference_note=reference_note,
        guarantee=BY_CONSTRUCTION,
    )


def standard_supplementary_operator() -> SupplementaryOperator:
    """Create the standard supplementary operator."""

    return from_fuzzy_negation(
        standard(),
        name="Standard supplementary operator",
        formula="I(x)=1-x for x>0; I(x)=x+1 for x<0; I(0)=0",
        reference_note="Supplementary operator induced by standard negation.",
    )


def yager_supplementary_operator(p: float) -> SupplementaryOperator:
    """Create the Yager supplementary operator from the article examples."""

    return from_fuzzy_negation(
        yager(p),
        name=f"Yager supplementary operator(p={p:g})",
        formula="I(x)=(1-x**p)**(1/p) for x>0; I(x)=-(1-(-x)**p)**(1/p) for x<0",
        reference_note="Yager supplementary operator used in the article/book examples.",
    )


def power_supplementary_operator(p: float) -> SupplementaryOperator:
    """Create supplementary operator induced by n_p(x)=1-x**p."""

    return from_fuzzy_negation(
        power(p),
        name=f"Power supplementary operator(p={p:g})",
        formula="I(x)=1-x**p for x>0; I(x)=(-x)**p-1 for x<0",
        reference_note="Power supplementary operator from the BFS examples.",
    )


def complement_power_supplementary_operator(p: float) -> SupplementaryOperator:
    """Create supplementary operator induced by n_p(x)=(1-x)**p."""

    return from_fuzzy_negation(
        complement_power(p),
        name=f"Complement-power supplementary operator(p={p:g})",
        formula="I(x)=(1-x)**p for x>0; I(x)=-((1+x)**p) for x<0",
        reference_note="Complement-power supplementary operator from the BFS examples.",
    )


def root_complement_supplementary_operator(m: float) -> SupplementaryOperator:
    """Create supplementary operator induced by n_m(x)=1-x**(1/m)."""

    return from_fuzzy_negation(
        root_complement(m),
        name=f"Root-complement supplementary operator(m={m:g})",
        formula="I(x)=1-x**(1/m) for x>0; I(x)=(-x)**(1/m)-1 for x<0",
        reference_note="Root-complement supplementary operator from the BFS examples.",
    )


def complement_root_supplementary_operator(m: float) -> SupplementaryOperator:
    """Create supplementary operator induced by n_m(x)=(1-x)**(1/m)."""

    return from_fuzzy_negation(
        complement_root(m),
        name=f"Complement-root supplementary operator(m={m:g})",
        formula="I(x)=(1-x)**(1/m) for x>0; I(x)=-((1+x)**(1/m)) for x<0",
        reference_note="Complement-root supplementary operator from the BFS examples.",
    )


def cosine_supplementary_operator() -> SupplementaryOperator:
    """Create the cosine supplementary operator."""

    return from_fuzzy_negation(
        cosine(),
        name="Cosine supplementary operator",
        formula="I(x)=0.5*(1+cos(pi*x)) for x>0; I(x)=-0.5*(1+cos(pi*x)) for x<0",
        reference_note="Cosine supplementary operator from the BFS examples.",
    )


def sugeno_supplementary_operator(lambda_: float) -> SupplementaryOperator:
    """Create the Sugeno supplementary operator."""

    return from_fuzzy_negation(
        sugeno(lambda_),
        name=f"Sugeno supplementary operator(lambda={lambda_:g})",
        formula="I(x)=(1-x)/(1+lambda*x) for x>0; I(x)=-(1+x)/(1-lambda*x) for x<0",
        reference_note="Sugeno supplementary operator from the BFS examples.",
    )


def known_supplementary_operators(
    sample_parameter: float = 2.0,
) -> list[SupplementaryOperator]:
    """Return representative supplementary operators from implemented examples."""

    return [
        standard_supplementary_operator(),
        yager_supplementary_operator(sample_parameter),
        power_supplementary_operator(sample_parameter),
        complement_power_supplementary_operator(sample_parameter),
        root_complement_supplementary_operator(sample_parameter),
        complement_root_supplementary_operator(sample_parameter),
        cosine_supplementary_operator(),
        sugeno_supplementary_operator(sample_parameter),
    ]


def supplementary_operators_summary() -> str:
    lines = [
        "Known implemented supplementary operators:",
        "- Standard: I(x)=1-x for x>0; I(x)=x+1 for x<0; I(0)=0",
        "- Yager: induced by n_p(x)=(1-x**p)**(1/p), p>0",
        "- Power: induced by n_p(x)=1-x**p, p>0",
        "- Complement power: induced by n_p(x)=(1-x)**p, p>0",
        "- Root complement: induced by n_m(x)=1-x**(1/m), m>0",
        "- Complement root: induced by n_m(x)=(1-x)**(1/m), m>0",
        "- Cosine: induced by n_cos(x)=0.5*(1+cos(pi*x))",
        "- Sugeno: induced by n_lambda(x)=(1-x)/(1+lambda*x), lambda>-1",
    ]
    return "\n".join(lines)


KNOWN_SUPPLEMENTARY_OPERATOR_FACTORIES = {
    "standard": standard_supplementary_operator,
    "yager": yager_supplementary_operator,
    "power": power_supplementary_operator,
    "complement_power": complement_power_supplementary_operator,
    "root_complement": root_complement_supplementary_operator,
    "complement_root": complement_root_supplementary_operator,
    "cosine": cosine_supplementary_operator,
    "sugeno": sugeno_supplementary_operator,
}


balanced_supplement = supplementary_operator
