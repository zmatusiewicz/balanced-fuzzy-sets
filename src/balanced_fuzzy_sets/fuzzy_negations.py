"""Basic fuzzy negations n:[0,1]->[0,1]."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, e, exp, isclose, isfinite, log, pi

from .operator_guarantees import (
    BY_CONSTRUCTION,
    SYMBOLICALLY_VERIFIED,
    normalize_guarantee,
)

NumberFunction = Callable[[float], float]
EPS = 1e-12
VALIDATION_EPS = 1e-9


def _unit_interval(x: float, name: str = "x") -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}")
    return x


def _positive(value: float, name: str = "p") -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _clamp_unit(x: float, name: str = "value") -> float:
    if not isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}.")
    if x < -VALIDATION_EPS or x > 1.0 + VALIDATION_EPS:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _validate_unit_output(x: float, name: str = "n(x)") -> float:
    return _clamp_unit(x, name)


@dataclass(frozen=True)
class FuzzyNegationProperties:
    """Properties checked empirically on a finite grid."""

    maps_unit_interval: bool
    boundary_conditions: bool
    empirically_decreasing_on_grid: bool
    empirically_strictly_decreasing_on_grid: bool
    empirically_involutive_on_grid: bool
    valid_fuzzy_negation: bool

    @property
    def decreasing(self) -> bool:
        return self.empirically_decreasing_on_grid

    @property
    def strictly_decreasing(self) -> bool:
        return self.empirically_strictly_decreasing_on_grid

    @property
    def involutive(self) -> bool:
        return self.empirically_involutive_on_grid


@dataclass(frozen=True)
class FuzzyNegation:
    """Concrete fuzzy negation n:[0,1]->[0,1]."""

    name: str
    formula: str
    n: NumberFunction
    parameter: float | None = None
    family: str | None = None
    is_strict: bool | None = None
    is_strong: bool | None = None
    reference_note: str = ""
    guarantee: str | None = None
    validate: bool = True
    validation_points: int = 401
    numerically_checked_properties: FuzzyNegationProperties = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=self.validate),
        )
        original_is_strict = self.is_strict
        original_is_strong = self.is_strong
        properties = self.check_properties()
        object.__setattr__(self, "numerically_checked_properties", properties)
        has_theoretical_guarantee = self.guarantee in {BY_CONSTRUCTION, SYMBOLICALLY_VERIFIED}
        object.__setattr__(
            self,
            "is_strict",
            original_is_strict if has_theoretical_guarantee or original_is_strict is False else None,
        )
        object.__setattr__(
            self,
            "is_strong",
            original_is_strong if has_theoretical_guarantee or original_is_strong is False else None,
        )
        if self.validate and not properties.valid_fuzzy_negation:
            failed = [
                name
                for name in (
                    "maps_unit_interval",
                    "boundary_conditions",
                    "empirically_decreasing_on_grid",
                )
                if not getattr(properties, name)
            ]
            raise ValueError(
                f"{self.name} is not a valid fuzzy negation; "
                f"failed properties: {', '.join(failed)}."
            )

    def __call__(self, x: float) -> float:
        x = _unit_interval(x)
        return _validate_unit_output(self.n(x), f"{self.name}({x})")

    @property
    def verified_properties(self) -> FuzzyNegationProperties:
        """Backward-compatible alias for numerically_checked_properties."""

        return self.numerically_checked_properties

    def check_properties(self) -> FuzzyNegationProperties:
        """Check fuzzy negation properties on a finite grid."""

        if self.validation_points < 3:
            raise ValueError("validation_points must be at least 3.")

        points = [
            index / (self.validation_points - 1)
            for index in range(self.validation_points)
        ]

        maps_unit_interval = True
        values = []
        for x in points:
            try:
                values.append(_validate_unit_output(self.n(x), f"{self.name}({x})"))
            except (ArithmeticError, ValueError, OverflowError):
                maps_unit_interval = False
                values.append(float("nan"))

        boundary_conditions = maps_unit_interval
        if maps_unit_interval:
            boundary_conditions = (
                abs(values[0] - 1.0) <= VALIDATION_EPS
                and abs(values[-1]) <= VALIDATION_EPS
            )

        decreasing = maps_unit_interval
        strictly_decreasing = maps_unit_interval
        previous = values[0]
        for value in values[1:]:
            if value > previous + VALIDATION_EPS:
                decreasing = False
            if value >= previous - VALIDATION_EPS:
                strictly_decreasing = False
            previous = value

        involutive = maps_unit_interval
        if maps_unit_interval:
            for x in points:
                value = _validate_unit_output(self.n(x), f"{self.name}({x})")
                try:
                    double_value = _validate_unit_output(
                        self.n(value),
                        f"{self.name}({self.name}({x}))",
                    )
                except (ArithmeticError, ValueError, OverflowError):
                    involutive = False
                    break
                if abs(double_value - x) > 1e-6:
                    involutive = False
                    break

        valid_fuzzy_negation = maps_unit_interval and boundary_conditions and decreasing
        return FuzzyNegationProperties(
            maps_unit_interval=maps_unit_interval,
            boundary_conditions=boundary_conditions,
            empirically_decreasing_on_grid=decreasing,
            empirically_strictly_decreasing_on_grid=strictly_decreasing,
            empirically_involutive_on_grid=involutive,
            valid_fuzzy_negation=valid_fuzzy_negation,
        )


def standard() -> FuzzyNegation:
    """Standard/Zadeh negation n(x)=1-x."""

    return FuzzyNegation(
        name="Standard",
        family="basic",
        formula="n(x) = 1 - x",
        n=lambda x: 1.0 - x,
        is_strict=True,
        is_strong=True,
        reference_note="Standard negation used in Lemma 1 of the attached paper.",
        guarantee=BY_CONSTRUCTION,
    )


def yager(p: float) -> FuzzyNegation:
    """Yager negation n_p(x)=(1-x**p)**(1/p), p>0."""

    _positive(p)
    return FuzzyNegation(
        name=f"Yager(p={p:g})",
        family="Yager",
        formula="n_p(x) = (1 - x**p)**(1/p), p > 0",
        n=lambda x: (1.0 - x**p) ** (1.0 / p),
        parameter=p,
        is_strict=True,
        is_strong=True,
        reference_note="Yager negation appears in the examples after Theorem 7.",
        guarantee=BY_CONSTRUCTION,
    )


def sugeno(lambda_: float) -> FuzzyNegation:
    """Sugeno negation n_lambda(x)=(1-x)/(1+lambda*x), lambda>-1."""

    if lambda_ <= -1.0:
        raise ValueError("Sugeno negation requires lambda_ > -1.")

    if isclose(lambda_, 0.0, abs_tol=EPS):
        base = standard()
        return FuzzyNegation(
            name="Sugeno(lambda=0)",
            family="Sugeno",
            formula=base.formula,
            n=base.n,
            parameter=0.0,
            is_strict=True,
            is_strong=True,
            reference_note="Limit case equal to the standard negation.",
            guarantee=BY_CONSTRUCTION,
        )

    return FuzzyNegation(
        name=f"Sugeno(lambda={lambda_:g})",
        family="Sugeno",
        formula="n_lambda(x) = (1 - x) / (1 + lambda*x), lambda > -1",
        n=lambda x: (1.0 - x) / (1.0 + lambda_ * x),
        parameter=lambda_,
        is_strict=True,
        is_strong=True,
        guarantee=BY_CONSTRUCTION,
    )


def power(p: float) -> FuzzyNegation:
    """Power negation n_p(x)=1-x**p, p>0."""

    _positive(p)
    return FuzzyNegation(
        name=f"Power(p={p:g})",
        family="Power",
        formula="n_p(x) = 1 - x**p, p > 0",
        n=lambda x: 1.0 - x**p,
        parameter=p,
        is_strict=True,
        is_strong=isclose(p, 1.0, abs_tol=EPS),
        reference_note="For p=2 this is a standard example of a strict but non-strong negation.",
        guarantee=BY_CONSTRUCTION,
    )


def complement_power(p: float) -> FuzzyNegation:
    """Complement-power negation n_p(x)=(1-x)**p, p>0."""

    _positive(p)
    return FuzzyNegation(
        name=f"ComplementPower(p={p:g})",
        family="Complement power",
        formula="n_p(x) = (1 - x)**p, p > 0",
        n=lambda x: (1.0 - x) ** p,
        parameter=p,
        is_strict=True,
        is_strong=isclose(p, 1.0, abs_tol=EPS),
        reference_note="Family n_Pm from the BFS book.",
        guarantee=BY_CONSTRUCTION,
    )


def root_complement(m: float) -> FuzzyNegation:
    """Root-complement negation n_m(x)=1-x**(1/m), m>0."""

    _positive(m, "m")
    return FuzzyNegation(
        name=f"RootComplement(m={m:g})",
        family="Root complement",
        formula="n_m(x) = 1 - x**(1/m), m > 0",
        n=lambda x: 1.0 - x ** (1.0 / m),
        parameter=m,
        is_strict=True,
        is_strong=isclose(m, 1.0, abs_tol=EPS),
        reference_note="Family n_rm from the BFS book.",
        guarantee=BY_CONSTRUCTION,
    )


def complement_root(m: float) -> FuzzyNegation:
    """Complement-root negation n_m(x)=(1-x)**(1/m), m>0."""

    _positive(m, "m")
    return FuzzyNegation(
        name=f"ComplementRoot(m={m:g})",
        family="Complement root",
        formula="n_m(x) = (1 - x)**(1/m), m > 0",
        n=lambda x: (1.0 - x) ** (1.0 / m),
        parameter=m,
        is_strict=True,
        is_strong=isclose(m, 1.0, abs_tol=EPS),
        reference_note="Family n_Rm from the BFS book.",
        guarantee=BY_CONSTRUCTION,
    )


def cosine() -> FuzzyNegation:
    """Cosine negation n(x)=0.5*(1+cos(pi*x))."""

    return FuzzyNegation(
        name="Cosine",
        family="Trigonometric",
        formula="n_cos(x) = 0.5 * (1 + cos(pi*x))",
        n=lambda x: 0.5 * (1.0 + cos(pi * x)),
        is_strict=True,
        is_strong=False,
        reference_note="Cosine negation from the BFS book; strict but not involutive.",
        guarantee=BY_CONSTRUCTION,
    )


def exponential_generated() -> FuzzyNegation:
    """Generated strong negation n(x)=log(e+1-exp(x))."""

    return generat_strong(
        phi=lambda x: (exp(x) - 1.0) / (e - 1.0),
        phi_inverse=lambda z: log(1.0 + z * (e - 1.0)),
        name="ExponentialGenerated",
        formula="n(x) = log(e + 1 - exp(x))",
        reference_note="Example generated by g(x)=exp(x)-1 in the BFS book.",
        guarantee=BY_CONSTRUCTION,
    )


def dombi_with_fixed_point(nu: float) -> FuzzyNegation:
    """
    Dombi/Sugeno negation with fixed point nu in (0,1).

    It is equivalent to Sugeno negation with lambda=((1-nu)/nu)**2 - 1.
    """

    if not 0.0 < nu < 1.0:
        raise ValueError(f"nu must be in (0, 1), got {nu!r}")

    coefficient = ((1.0 - nu) / nu) ** 2

    def n(x: float) -> float:
        if x == 1.0:
            return 0.0
        return 1.0 / (1.0 + coefficient * x / (1.0 - x))

    return FuzzyNegation(
        name=f"Dombi(nu={nu:g})",
        family="Dombi",
        formula="n_nu(x)=1/(1+((1-nu)/nu)**2 * x/(1-x)); n_nu(1)=0",
        n=n,
        parameter=nu,
        is_strict=True,
        is_strong=True,
        reference_note="Dombi form of negation, also expressible as a Sugeno negation.",
        guarantee=BY_CONSTRUCTION,
    )


def _validate_increasing_automorphism(
    phi: NumberFunction,
    phi_inverse: NumberFunction,
    *,
    validation_points: int = 401,
) -> None:
    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    points = [
        index / (validation_points - 1)
        for index in range(validation_points)
    ]

    previous = None
    for x in points:
        value = _validate_unit_output(phi(x), f"phi({x})")
        if previous is not None and value <= previous + VALIDATION_EPS:
            raise ValueError("phi must be strictly increasing on [0,1].")
        previous = value

        roundtrip = _validate_unit_output(phi_inverse(value), f"phi_inverse(phi({x}))")
        if abs(roundtrip - x) > 1e-6:
            raise ValueError(
                f"phi_inverse must invert phi; phi_inverse(phi({x}))={roundtrip!r}."
            )

    previous = None
    for y in points:
        value = _validate_unit_output(phi_inverse(y), f"phi_inverse({y})")
        if previous is not None and value <= previous + VALIDATION_EPS:
            raise ValueError("phi_inverse must be strictly increasing on [0,1].")
        previous = value

        roundtrip = _validate_unit_output(phi(value), f"phi(phi_inverse({y}))")
        if abs(roundtrip - y) > 1e-6:
            raise ValueError(
                f"phi must invert phi_inverse; phi(phi_inverse({y}))={roundtrip!r}."
            )


def generat_strong(
    phi: NumberFunction,
    phi_inverse: NumberFunction,
    *,
    name: str = "Generat strong",
    formula: str = "n(x)=phi^{-1}(1-phi(x))",
    reference_note: str = "",
    guarantee: str | None = None,
) -> FuzzyNegation:
    """Generat strong fuzzy negation by an increasing automorphism of [0,1]."""

    if abs(phi(0.0)) > EPS or abs(phi(1.0) - 1.0) > EPS:
        raise ValueError("phi must satisfy phi(0)=0 and phi(1)=1.")
    _validate_increasing_automorphism(phi, phi_inverse)

    def n(x: float) -> float:
        return phi_inverse(1.0 - _unit_interval(phi(x), "phi(x)"))

    return FuzzyNegation(
        name=name,
        family="Generated strong",
        formula=formula,
        n=n,
        is_strict=True,
        is_strong=True,
        reference_note=reference_note,
        guarantee=guarantee,
    )


generated_strong = generat_strong


def godel() -> FuzzyNegation:
    """Goedel negation: n(0)=1 and n(x)=0 for x>0."""

    return FuzzyNegation(
        name="Goedel",
        family="basic",
        formula="n(0) = 1; n(x) = 0 for x > 0",
        n=lambda x: 1.0 if x == 0.0 else 0.0,
        is_strict=False,
        is_strong=False,
        reference_note="Non-strict basic fuzzy negation.",
        guarantee=BY_CONSTRUCTION,
    )


def intuitionistic() -> FuzzyNegation:
    """Intuitionistic/least negation n_0."""

    return FuzzyNegation(
        name="Intuitionistic",
        family="basic",
        formula="n_0(0) = 1; n_0(x) = 0 for x > 0",
        n=lambda x: 1.0 if x == 0.0 else 0.0,
        is_strict=False,
        is_strong=False,
        reference_note="Least fuzzy negation from the BFS book.",
        guarantee=BY_CONSTRUCTION,
    )


def greatest() -> FuzzyNegation:
    """Greatest pointwise fuzzy negation: n(x)=1 for x<1 and n(1)=0."""

    return FuzzyNegation(
        name="Greatest",
        family="basic",
        formula="n(x) = 1 for x < 1; n(1) = 0",
        n=lambda x: 0.0 if x == 1.0 else 1.0,
        is_strict=False,
        is_strong=False,
        reference_note="Greatest pointwise fuzzy negation under the standard order.",
        guarantee=BY_CONSTRUCTION,
    )


def dual_intuitionistic() -> FuzzyNegation:
    """Dual intuitionistic/greatest negation n_1."""

    base = greatest()
    return FuzzyNegation(
        name="DualIntuitionistic",
        family=base.family,
        formula="n_1(x) = 1 for x < 1; n_1(1) = 0",
        n=base.n,
        is_strict=base.is_strict,
        is_strong=base.is_strong,
        reference_note="Greatest fuzzy negation from the BFS book.",
        guarantee=base.guarantee,
    )


def least() -> FuzzyNegation:
    """Deprecated alias for greatest(); kept for backward compatibility."""

    base = greatest()
    return FuzzyNegation(
        name="Greatest",
        family=base.family,
        formula=base.formula,
        n=base.n,
        is_strict=base.is_strict,
        is_strong=base.is_strong,
        reference_note="Deprecated alias: use greatest() or dual_intuitionistic().",
        guarantee=base.guarantee,
    )


def standard_negation(x: float) -> float:
    """Evaluate the standard fuzzy negation n(x)=1-x."""

    return standard()(x)


def yager_negation(x: float, p: float) -> float:
    """Evaluate Yager negation for a single value."""

    return yager(p)(x)


def sugeno_negation(x: float, lambda_: float) -> float:
    """Evaluate Sugeno negation for a single value."""

    return sugeno(lambda_)(x)


def power_negation(x: float, p: float) -> float:
    """Evaluate power negation for a single value."""

    return power(p)(x)


def dombi_negation(x: float, nu: float) -> float:
    """Evaluate Dombi/Sugeno negation with fixed point nu for a single value."""

    return dombi_with_fixed_point(nu)(x)


def complement_power_negation(x: float, p: float) -> float:
    """Evaluate complement-power negation for a single value."""

    return complement_power(p)(x)


def root_complement_negation(x: float, m: float) -> float:
    """Evaluate root-complement negation for a single value."""

    return root_complement(m)(x)


def complement_root_negation(x: float, m: float) -> float:
    """Evaluate complement-root negation for a single value."""

    return complement_root(m)(x)


def cosine_negation(x: float) -> float:
    """Evaluate cosine negation for a single value."""

    return cosine()(x)


def known_basic_negations(sample_parameter: float = 2.0) -> list[FuzzyNegation]:
    """Return representative basic fuzzy negations implemented in this module."""

    return [
        standard(),
        yager(sample_parameter),
        sugeno(sample_parameter),
        power(sample_parameter),
        complement_power(sample_parameter),
        root_complement(sample_parameter),
        complement_root(sample_parameter),
        cosine(),
        exponential_generated(),
        dombi_with_fixed_point(0.5),
        intuitionistic(),
        greatest(),
        dual_intuitionistic(),
    ]


def negations_summary() -> str:
    lines = [
        "Known implemented fuzzy negations:",
        "- Standard/Zadeh: n(x)=1-x",
        "- Yager: n_p(x)=(1-x**p)**(1/p), p>0",
        "- Sugeno: n_lambda(x)=(1-x)/(1+lambda*x), lambda>-1",
        "- Power: n_p(x)=1-x**p, p>0",
        "- Complement power: n_p(x)=(1-x)**p, p>0",
        "- Root complement: n_m(x)=1-x**(1/m), m>0",
        "- Complement root: n_m(x)=(1-x)**(1/m), m>0",
        "- Cosine: n_cos(x)=0.5*(1+cos(pi*x))",
        "- Exponential generated: n(x)=log(e+1-exp(x))",
        "- Dombi/Sugeno fixed point: n_nu(x)=1/(1+((1-nu)/nu)**2*x/(1-x)), nu in (0,1)",
        "- Generat strong: n(x)=phi^{-1}(1-phi(x)) for an increasing automorphism phi",
        "- Intuitionistic: n_0(0)=1 and n_0(x)=0 for x>0",
        "- Greatest: n_1(x)=1 for x<1 and n_1(1)=0",
        "- Dual intuitionistic: n_1(x)=1 for x<1 and n_1(1)=0",
    ]
    return "\n".join(lines)


KNOWN_NEGATION_FACTORIES = {
    "standard": standard,
    "zadeh": standard,
    "yager": yager,
    "sugeno": sugeno,
    "power": power,
    "complement_power": complement_power,
    "root_complement": root_complement,
    "complement_root": complement_root,
    "cosine": cosine,
    "exponential_generated": exponential_generated,
    "dombi": dombi_with_fixed_point,
    "intuitionistic": intuitionistic,
    "greatest": greatest,
    "dual_intuitionistic": dual_intuitionistic,
    "least": least,
}
