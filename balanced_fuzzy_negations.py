"""Concrete balanced fuzzy negations N_B:[-1,1]->[-1,1]."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, isfinite, pi, sqrt
from typing import Callable

from operator_guarantees import BY_CONSTRUCTION, SYMBOLICALLY_VERIFIED, normalize_guarantee


NumberFunction = Callable[[float], float]
EPS = 1e-9
SURJECTIVITY_GUARANTEED_BY_CONTINUITY = "guaranteed_by_continuity"
SURJECTIVITY_FAILED_ENDPOINTS = "failed_endpoints"
SURJECTIVITY_UNKNOWN = "unknown"
PROPERTY_NAMES = frozenset(
    {
        "maps_balanced_interval",
        "fixed_zero",
        "decreasing",
        "strictly_decreasing",
        "sign_reversing",
        "endpoint_surjective",
        "bijective",
        "involutive",
        "balanced_negation",
        "strict",
        "strong",
        "valid_balanced_negation",
    }
)


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


def _validate_balanced_output(value: float, name: str = "N(x)") -> float:
    value = _clamp_balanced(value, name)
    if abs(value) <= EPS:
        return 0.0
    return value


@dataclass(frozen=True)
class BalancedNegationProperties:
    """Properties checked empirically on a finite grid."""

    maps_balanced_interval: bool
    fixed_zero: bool
    empirically_decreasing_on_grid: bool
    empirically_strictly_decreasing_on_grid: bool
    sign_reversing: bool
    endpoint_surjective: bool
    empirically_injective_on_grid: bool
    surjectivity_status: str
    empirically_bijective_on_grid: bool | None
    empirically_involutive_on_grid: bool
    is_balanced_negation: bool
    empirically_strict_on_grid: bool | None
    empirically_strong_on_grid: bool

    @property
    def decreasing(self) -> bool:
        return self.empirically_decreasing_on_grid

    @property
    def strictly_decreasing(self) -> bool:
        return self.empirically_strictly_decreasing_on_grid

    @property
    def bijective(self) -> bool:
        return self.empirically_bijective_on_grid is True

    @property
    def involutive(self) -> bool:
        return self.empirically_involutive_on_grid

    @property
    def valid_balanced_negation(self) -> bool:
        return self.is_balanced_negation


@dataclass(frozen=True)
class BalancedFuzzyNegation:
    """Concrete balanced fuzzy negation with N_B(0)=0."""

    name: str
    formula: str
    n: NumberFunction
    is_strict: bool | None = None
    is_strong: bool | None = None
    is_continuous: bool | None = None
    reference_note: str = ""
    guarantee: str | None = None
    validate: bool = True
    validation_points: int = 401
    claimed_properties: frozenset[str] = field(default_factory=frozenset)
    numerically_checked_properties: BalancedNegationProperties = field(init=False)

    def __call__(self, x: float) -> float:
        x = _balanced_interval(x)
        return _validate_balanced_output(self.n(x), f"{self.name}({x})")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=self.validate),
        )
        original_is_strict = self.is_strict
        original_is_strong = self.is_strong
        original_is_continuous = self.is_continuous
        claimed = set(self.claimed_properties)
        if original_is_strict is True:
            claimed.add("strictly_decreasing")
            claimed.add("bijective")
        if original_is_strong is True:
            claimed.add("involutive")
        unknown = claimed - PROPERTY_NAMES
        if unknown:
            raise ValueError(f"Unknown claimed properties: {sorted(unknown)!r}.")
        object.__setattr__(self, "claimed_properties", frozenset(claimed))

        properties = self.check_properties()
        object.__setattr__(self, "numerically_checked_properties", properties)
        has_theoretical_guarantee = self.guarantee in {BY_CONSTRUCTION, SYMBOLICALLY_VERIFIED}
        has_continuity_guarantee = (
            original_is_continuous is True and has_theoretical_guarantee
        )
        object.__setattr__(
            self,
            "is_continuous",
            True if has_continuity_guarantee else original_is_continuous,
        )
        object.__setattr__(
            self,
            "is_strict",
            original_is_strict
            if (
                original_is_strict is False
                or (has_theoretical_guarantee and properties.empirically_strict_on_grid is True)
            )
            else None,
        )
        object.__setattr__(
            self,
            "is_strong",
            original_is_strong if has_theoretical_guarantee or original_is_strong is False else None,
        )

        if self.validate and not properties.valid_balanced_negation:
            failed = [
                name
                for name in (
                    "maps_balanced_interval",
                    "fixed_zero",
                    "empirically_decreasing_on_grid",
                    "sign_reversing",
                )
                if not getattr(properties, name)
            ]
            raise ValueError(
                f"{self.name} is not a valid balanced fuzzy negation; "
                f"failed properties: {', '.join(failed)}."
            )
        if (
            self.validate
            and original_is_strict is True
            and properties.empirically_strict_on_grid is False
        ):
            raise ValueError(
                f"{self.name} is declared strict, but numerical checks did not "
                "confirm balanced negation plus bijectivity under the stated "
                "continuity/theoretical guarantees."
            )
        if self.validate and original_is_strong is True and not properties.empirically_strong_on_grid:
            raise ValueError(
                f"{self.name} is declared strong, but numerical checks did not "
                "confirm balanced negation plus involutivity on the validation grid."
            )

    def validate_properties(self) -> None:
        """Validate core balanced fuzzy negation properties on a finite grid."""

        properties = self.check_properties()
        if not properties.is_balanced_negation:
            raise ValueError(f"{self.name} is not a valid balanced fuzzy negation.")

    @property
    def verified_properties(self) -> BalancedNegationProperties:
        """Backward-compatible alias for numerically_checked_properties."""

        return self.numerically_checked_properties

    def check_properties(self) -> BalancedNegationProperties:
        """Return numerically checked properties on a finite grid."""

        if self.validation_points < 3:
            raise ValueError("validation_points must be at least 3.")

        points = [
            -1.0 + 2.0 * index / (self.validation_points - 1)
            for index in range(self.validation_points)
        ]
        maps_balanced_interval = True
        values = []
        for x in points:
            try:
                values.append(_validate_balanced_output(self.n(x), f"{self.name}({x})"))
            except ValueError:
                maps_balanced_interval = False
                values.append(float("nan"))

        try:
            zero_value = _validate_balanced_output(self.n(0.0), f"{self.name}(0)")
            fixed_zero = abs(zero_value) <= EPS
        except ValueError:
            fixed_zero = False

        try:
            left = _validate_balanced_output(self.n(-1.0), f"{self.name}(-1)")
            right = _validate_balanced_output(self.n(1.0), f"{self.name}(1)")
            endpoint_surjective = abs(left - 1.0) <= EPS and abs(right + 1.0) <= EPS
        except ValueError:
            endpoint_surjective = False

        decreasing = maps_balanced_interval
        strictly_decreasing = maps_balanced_interval
        previous_value = values[0]
        for x, value in zip(points[1:], values[1:]):
            if value > previous_value + EPS:
                decreasing = False
            if value >= previous_value - EPS:
                strictly_decreasing = False
            previous_value = value

        sign_reversing = maps_balanced_interval
        for x, value in zip(points, values):
            if x * value > EPS:
                sign_reversing = False
                break

        rounded_values = {round(value, 12) for value in values}
        injective_on_grid = len(rounded_values) == len(values)

        involutive = maps_balanced_interval
        if maps_balanced_interval:
            for x in points:
                value = _validate_balanced_output(self.n(x), f"{self.name}({x})")
                try:
                    involutive_value = _validate_balanced_output(
                        self.n(value),
                        f"{self.name}({self.name}({x}))",
                    )
                except ValueError:
                    involutive = False
                    break
                if abs(involutive_value - x) > 1e-6:
                    involutive = False
                    break

        is_balanced_negation = (
            maps_balanced_interval
            and fixed_zero
            and decreasing
            and sign_reversing
        )
        has_theoretical_guarantee = self.guarantee in {BY_CONSTRUCTION, SYMBOLICALLY_VERIFIED}
        continuity_guaranteed = self.is_continuous is True and has_theoretical_guarantee
        if not endpoint_surjective:
            surjectivity_status = SURJECTIVITY_FAILED_ENDPOINTS
            bijective: bool | None = False
        elif continuity_guaranteed and strictly_decreasing:
            surjectivity_status = SURJECTIVITY_GUARANTEED_BY_CONTINUITY
            bijective = is_balanced_negation and injective_on_grid
        else:
            surjectivity_status = SURJECTIVITY_UNKNOWN
            bijective = None
        empirically_strict_on_grid = is_balanced_negation and bijective
        empirically_strong_on_grid = is_balanced_negation and involutive

        return BalancedNegationProperties(
            maps_balanced_interval=maps_balanced_interval,
            fixed_zero=fixed_zero,
            empirically_decreasing_on_grid=decreasing,
            empirically_strictly_decreasing_on_grid=strictly_decreasing,
            sign_reversing=sign_reversing,
            endpoint_surjective=endpoint_surjective,
            empirically_injective_on_grid=injective_on_grid,
            surjectivity_status=surjectivity_status,
            empirically_bijective_on_grid=bijective,
            empirically_involutive_on_grid=involutive,
            is_balanced_negation=is_balanced_negation,
            empirically_strict_on_grid=empirically_strict_on_grid,
            empirically_strong_on_grid=empirically_strong_on_grid,
        )


def reversal() -> BalancedFuzzyNegation:
    """Balanced reversal operation N(x)=-x from the attached paper."""

    return BalancedFuzzyNegation(
        name="Reversal",
        formula="N(x) = -x",
        n=lambda x: -x,
        is_strict=True,
        is_strong=True,
        is_continuous=True,
        reference_note="Reversal operation used in the definition of balanced operations.",
        guarantee=BY_CONSTRUCTION,
    )


def square() -> BalancedFuzzyNegation:
    """Square balanced fuzzy negation from Table 3.6."""

    def n(x: float) -> float:
        if x >= 0.0:
            return -(x**2)
        return x**2

    return BalancedFuzzyNegation(
        name="Square",
        formula="N_I(x)=-x**2 for x>=0; N_I(x)=x**2 for x<0",
        n=n,
        is_strict=True,
        is_strong=False,
        is_continuous=True,
        reference_note="Balanced fuzzy negation from Table 3.6; the table labels it as supplementary.",
        guarantee=BY_CONSTRUCTION,
    )


def cube() -> BalancedFuzzyNegation:
    """Cube balanced fuzzy negation from Table 3.6."""

    return BalancedFuzzyNegation(
        name="Cube",
        formula="N_I(x) = -x**3",
        n=lambda x: -(x**3),
        is_strict=True,
        is_strong=False,
        is_continuous=True,
        reference_note="Balanced fuzzy negation from Table 3.6; the table labels it as supplementary.",
        guarantee=BY_CONSTRUCTION,
    )


def square_root() -> BalancedFuzzyNegation:
    """Square-root balanced fuzzy negation from Table 3.6."""

    def n(x: float) -> float:
        if x >= 0.0:
            return -sqrt(x)
        return sqrt(-x)

    return BalancedFuzzyNegation(
        name="SquareRoot",
        formula="N_I(x)=-sqrt(x) for x>=0; N_I(x)=sqrt(-x) for x<0",
        n=n,
        is_strict=True,
        is_strong=False,
        is_continuous=True,
        reference_note="Balanced fuzzy negation from Table 3.6; the table labels it as supplementary.",
        guarantee=BY_CONSTRUCTION,
    )


def cosine() -> BalancedFuzzyNegation:
    """Cosine balanced fuzzy negation from Table 3.6."""

    def n(x: float) -> float:
        if x >= 0.0:
            return 0.5 * (cos(pi * x) - 1.0)
        return 0.5 * (1.0 - cos(pi * x))

    return BalancedFuzzyNegation(
        name="Cosine",
        formula="N_I(x)=(cos(pi*x)-1)/2 for x>=0; N_I(x)=(1-cos(pi*x))/2 for x<0",
        n=n,
        is_strict=True,
        is_strong=False,
        is_continuous=True,
        reference_note="Balanced fuzzy negation from Table 3.6; the table labels it as supplementary.",
        guarantee=BY_CONSTRUCTION,
    )


def sugeno(lambda_: float) -> BalancedFuzzyNegation:
    """Sugeno balanced fuzzy negation from Table 3.6."""

    if lambda_ <= -1.0:
        raise ValueError("Sugeno balanced fuzzy negation requires lambda_ > -1.")

    def n(x: float) -> float:
        if x > 0.0:
            return (1.0 - x) / (1.0 + lambda_ * x) - 1.0
        if x < 0.0:
            return 1.0 - (1.0 + x) / (1.0 - lambda_ * x)
        return 0.0

    return BalancedFuzzyNegation(
        name=f"Sugeno(lambda={lambda_:g})",
        formula=(
            "N_I(x)=(1-x)/(1+lambda*x)-1 for x>0; "
            "N_I(x)=1-(1+x)/(1-lambda*x) for x<0; N_I(0)=0"
        ),
        n=n,
        is_strict=True,
        is_strong=False,
        is_continuous=True,
        reference_note="Balanced fuzzy negation from Table 3.6; the table labels it as supplementary.",
        guarantee=BY_CONSTRUCTION,
    )


def balanced_reversal(x: float) -> float:
    """Evaluate the balanced reversal operation N(x)=-x."""

    return reversal()(x)


def square_negation(x: float) -> float:
    """Evaluate square balanced fuzzy negation."""

    return square()(x)


def cube_negation(x: float) -> float:
    """Evaluate cube balanced fuzzy negation."""

    return cube()(x)


def square_root_negation(x: float) -> float:
    """Evaluate square-root balanced fuzzy negation."""

    return square_root()(x)


def cosine_negation(x: float) -> float:
    """Evaluate cosine balanced fuzzy negation."""

    return cosine()(x)


def sugeno_negation(x: float, lambda_: float) -> float:
    """Evaluate Sugeno balanced fuzzy negation."""

    return sugeno(lambda_)(x)


def known_balanced_fuzzy_negations(
    sample_parameter: float = 2.0,
) -> list[BalancedFuzzyNegation]:
    """Return representative balanced fuzzy negations from Table 3.6."""

    return [
        reversal(),
        square(),
        cube(),
        square_root(),
        cosine(),
        sugeno(sample_parameter),
    ]


def balanced_fuzzy_negations_summary() -> str:
    lines = [
        "Known implemented balanced fuzzy negations from Table 3.6:",
        "- Reversal: N(x)=-x",
        "- Square: N_I(x)=-x**2 for x>=0; N_I(x)=x**2 for x<0",
        "- Cube: N_I(x)=-x**3",
        "- Square root: N_I(x)=-sqrt(x) for x>=0; N_I(x)=sqrt(-x) for x<0",
        "- Cosine: N_I(x)=(cos(pi*x)-1)/2 for x>=0; N_I(x)=(1-cos(pi*x))/2 for x<0",
        "- Sugeno: induced by n_lambda(x)=(1-x)/(1+lambda*x), lambda>-1",
    ]
    return "\n".join(lines)


KNOWN_BALANCED_FUZZY_NEGATION_FACTORIES = {
    "reversal": reversal,
    "square": square,
    "cube": cube,
    "square_root": square_root,
    "cosine": cosine,
    "sugeno": sugeno,
}
