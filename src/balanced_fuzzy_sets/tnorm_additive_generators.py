"""
Additive generators for triangular norms (t-norms).

The additive-generator construction uses a decreasing function f with f(1)=0:

    T(x, y) = f_inv(min(f(0+), f(x) + f(y)))

where f_inv is the ordinary inverse for strict generators and the
pseudo-inverse for nilpotent generators.

Included literature families:
    - Lukasiewicz
    - Product
    - Drastic
    - Schweizer-Sklar
    - Hamacher
    - Frank
    - Yager
    - Aczel-Alsina
    - Dombi
    - Sugeno-Weber

Notes:
    - Positive scalar multiples of any additive generator define the same
      t-norm.
    - The minimum t-norm has no additive generator because it has non-trivial
      idempotents.
    - General ordinal sums are fundamental in the theory of continuous
      t-norms, but they are not, in general, represented by one additive
      generator on the whole unit interval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp, expm1, inf, isclose, isfinite, log, log1p

from .operator_guarantees import BY_CONSTRUCTION, normalize_guarantee

NumberFunction = Callable[[float], float]


EPS = 1e-12
VALIDATION_EPS = 1e-8
NEAR_LIMIT_EPS = 1e-8
MAX_EXP_ARGUMENT = 709.0


def _unit(x: float, name: str = "x") -> float:
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


@dataclass(frozen=True)
class AdditiveGeneratorProperties:
    """Properties checked empirically on finite grids."""

    maps_to_nonnegative_extended_reals: bool
    empirically_decreasing_on_grid: bool
    f_one_zero: bool
    f0_plus_consistent: bool
    inverse_consistent: bool
    t_norm_maps_unit_square: bool
    empirically_t_norm_commutative_on_grid: bool
    empirically_t_norm_associative_on_grid: bool
    empirically_t_norm_non_decreasing_on_grid: bool
    t_norm_boundary_conditions: bool
    valid_additive_generator: bool

    @property
    def decreasing(self) -> bool:
        return self.empirically_decreasing_on_grid

    @property
    def t_norm_commutative(self) -> bool:
        return self.empirically_t_norm_commutative_on_grid

    @property
    def t_norm_associative(self) -> bool:
        return self.empirically_t_norm_associative_on_grid

    @property
    def t_norm_non_decreasing(self) -> bool:
        return self.empirically_t_norm_non_decreasing_on_grid


@dataclass(frozen=True)
class AdditiveGenerator:
    """A concrete additive generator and its inverse/pseudo-inverse."""

    name: str
    formula: str
    f: NumberFunction
    finv: NumberFunction
    f0_plus: float
    parameter: float | None = None
    family: str | None = None
    reference_note: str = ""
    guarantee: str | None = None
    validate: bool = True
    validation_points: int = 101
    numerically_checked_properties: AdditiveGeneratorProperties = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=self.validate),
        )
        properties = self.check_properties()
        object.__setattr__(self, "numerically_checked_properties", properties)
        if self.validate and not properties.valid_additive_generator:
            failed = [
                name
                for name in (
                    "maps_to_nonnegative_extended_reals",
                    "empirically_decreasing_on_grid",
                    "f_one_zero",
                    "f0_plus_consistent",
                    "inverse_consistent",
                    "t_norm_maps_unit_square",
                    "empirically_t_norm_commutative_on_grid",
                    "empirically_t_norm_associative_on_grid",
                    "empirically_t_norm_non_decreasing_on_grid",
                    "t_norm_boundary_conditions",
                )
                if not getattr(properties, name)
            ]
            raise ValueError(
                f"{self.name} is not a valid additive generator; "
                f"failed properties: {', '.join(failed)}."
            )

    def __call__(self, x: float) -> float:
        return self.f(_unit(x))

    def inverse(self, s: float) -> float:
        if s < 0.0:
            raise ValueError(f"s must be non-negative, got {s!r}")
        return _clamp_unit(self.finv(s), f"{self.name}.finv({s})")

    def t_norm(self, x: float, y: float) -> float:
        x = _unit(x, "x")
        y = _unit(y, "y")
        s = self.f(x) + self.f(y)
        if self.f0_plus != inf:
            s = min(self.f0_plus, s)
        return self.inverse(s)

    @property
    def verified_properties(self) -> AdditiveGeneratorProperties:
        """Backward-compatible alias for numerically_checked_properties."""

        return self.numerically_checked_properties

    def check_properties(self) -> AdditiveGeneratorProperties:
        """Check generator assumptions and t-norm axioms on finite grids."""

        if self.validation_points < 3:
            raise ValueError("validation_points must be at least 3.")

        points = [
            index / (self.validation_points - 1)
            for index in range(self.validation_points)
        ]
        positive_points = points[1:]

        maps_to_nonnegative_extended_reals = True
        f_values: dict[float, float] = {}
        for x in points:
            try:
                value = self.f(x)
                if value < -VALIDATION_EPS or (not isfinite(value) and value != inf):
                    maps_to_nonnegative_extended_reals = False
                f_values[x] = value
            except (ArithmeticError, ValueError, OverflowError):
                maps_to_nonnegative_extended_reals = False
                f_values[x] = float("nan")

        f_one_zero = abs(f_values[1.0]) <= VALIDATION_EPS

        decreasing = maps_to_nonnegative_extended_reals
        previous = f_values[0.0]
        for x in positive_points:
            value = f_values[x]
            if value > previous + VALIDATION_EPS:
                decreasing = False
                break
            previous = value

        f0_value = f_values[0.0]
        if self.f0_plus == inf:
            f0_plus_consistent = f0_value == inf
        else:
            f0_plus_consistent = (
                isfinite(self.f0_plus)
                and self.f0_plus >= -VALIDATION_EPS
                and abs(f0_value - self.f0_plus) <= 1e-6
            )

        inverse_consistent = maps_to_nonnegative_extended_reals
        for x in positive_points:
            value = f_values[x]
            if value == inf:
                continue
            try:
                inverse_value = self.inverse(value)
            except (ArithmeticError, ValueError, OverflowError):
                inverse_consistent = False
                break
            if abs(inverse_value - x) > 1e-6:
                inverse_consistent = False
                break

        if inverse_consistent:
            sampled_s = _sample_inverse_arguments(f_values, self.f0_plus)
            for s in sampled_s:
                try:
                    x = self.inverse(s)
                    reconstructed = self.f(x)
                except (ArithmeticError, ValueError, OverflowError):
                    inverse_consistent = False
                    break
                if (s < self.f0_plus or self.f0_plus == inf) and abs(
                    reconstructed - s
                ) > 1e-5:
                    inverse_consistent = False
                    break

        t_points = _validation_subset(points, limit=13)
        t_norm_maps_unit_square = True
        t_values: dict[tuple[float, float], float] = {}
        for x in t_points:
            for y in t_points:
                try:
                    value = self.t_norm(x, y)
                    if value < -VALIDATION_EPS or value > 1.0 + VALIDATION_EPS:
                        t_norm_maps_unit_square = False
                    t_values[(x, y)] = value
                except (ArithmeticError, ValueError, OverflowError):
                    t_norm_maps_unit_square = False
                    t_values[(x, y)] = float("nan")

        t_norm_commutative = t_norm_maps_unit_square
        for x in t_points:
            for y in t_points:
                if abs(t_values[(x, y)] - t_values[(y, x)]) > 1e-6:
                    t_norm_commutative = False
                    break

        t_norm_boundary_conditions = t_norm_maps_unit_square
        for x in t_points:
            if (
                abs(self.t_norm(x, 1.0) - x) > 1e-6
                or abs(self.t_norm(1.0, x) - x) > 1e-6
                or abs(self.t_norm(x, 0.0)) > 1e-6
                or abs(self.t_norm(0.0, x)) > 1e-6
            ):
                t_norm_boundary_conditions = False
                break

        t_norm_non_decreasing = t_norm_maps_unit_square
        for x1 in t_points:
            for x2 in t_points:
                if x1 > x2:
                    continue
                for y1 in t_points:
                    for y2 in t_points:
                        if y1 > y2:
                            continue
                        if t_values[(x1, y1)] > t_values[(x2, y2)] + 1e-6:
                            t_norm_non_decreasing = False
                            break

        associativity_points = _validation_subset(points, limit=9)
        t_norm_associative = t_norm_maps_unit_square
        for x in associativity_points:
            for y in associativity_points:
                for z in associativity_points:
                    left = self.t_norm(self.t_norm(x, y), z)
                    right = self.t_norm(x, self.t_norm(y, z))
                    if abs(left - right) > 1e-6:
                        t_norm_associative = False
                        break

        valid_additive_generator = (
            maps_to_nonnegative_extended_reals
            and decreasing
            and f_one_zero
            and f0_plus_consistent
            and inverse_consistent
            and t_norm_maps_unit_square
            and t_norm_commutative
            and t_norm_associative
            and t_norm_non_decreasing
            and t_norm_boundary_conditions
        )

        return AdditiveGeneratorProperties(
            maps_to_nonnegative_extended_reals=maps_to_nonnegative_extended_reals,
            empirically_decreasing_on_grid=decreasing,
            f_one_zero=f_one_zero,
            f0_plus_consistent=f0_plus_consistent,
            inverse_consistent=inverse_consistent,
            t_norm_maps_unit_square=t_norm_maps_unit_square,
            empirically_t_norm_commutative_on_grid=t_norm_commutative,
            empirically_t_norm_associative_on_grid=t_norm_associative,
            empirically_t_norm_non_decreasing_on_grid=t_norm_non_decreasing,
            t_norm_boundary_conditions=t_norm_boundary_conditions,
            valid_additive_generator=valid_additive_generator,
        )


def lukasiewicz() -> AdditiveGenerator:
    return AdditiveGenerator(
        name="Lukasiewicz",
        family="basic",
        formula="f(x) = 1 - x",
        f=lambda x: 1.0 - x,
        finv=lambda s: max(0.0, 1.0 - s),
        f0_plus=1.0,
        reference_note="Nilpotent basic t-norm.",
        guarantee=BY_CONSTRUCTION,
    )


def product() -> AdditiveGenerator:
    return AdditiveGenerator(
        name="Product",
        family="basic",
        formula="f(x) = -log(x), f(0)=+inf",
        f=lambda x: inf if x == 0.0 else -log(x),
        finv=lambda s: exp(-s),
        f0_plus=inf,
        reference_note="Strict basic t-norm.",
        guarantee=BY_CONSTRUCTION,
    )


def drastic() -> AdditiveGenerator:
    def f(x: float) -> float:
        return 0.0 if x == 1.0 else 2.0 - x

    def finv(s: float) -> float:
        if s == 0.0:
            return 1.0
        if s >= 2.0:
            return 0.0
        return 2.0 - s

    return AdditiveGenerator(
        name="Drastic",
        family="basic",
        formula="f(1)=0, f(x)=2-x for 0 <= x < 1",
        f=f,
        finv=finv,
        f0_plus=2.0,
        reference_note="Non-continuous Archimedean t-norm.",
        guarantee=BY_CONSTRUCTION,
    )


def schweizer_sklar(p: float) -> AdditiveGenerator:
    if p in (-inf, inf):
        raise ValueError("Schweizer-Sklar additive generator is defined for finite p.")
    if isclose(p, 0.0, abs_tol=EPS):
        base = product()
        return AdditiveGenerator(
            name="Schweizer-Sklar(p=0)",
            family="Schweizer-Sklar",
            formula=base.formula,
            f=base.f,
            finv=base.finv,
            f0_plus=base.f0_plus,
            parameter=0.0,
            reference_note="Limit case equal to the product t-norm.",
            guarantee=BY_CONSTRUCTION,
        )

    f0_plus = 1.0 / p if p > 0.0 else inf

    def f(x: float) -> float:
        if x == 0.0 and p < 0.0:
            return inf
        return (1.0 - x**p) / p

    def finv(s: float) -> float:
        if p > 0.0 and s >= f0_plus:
            return 0.0
        return (1.0 - p * s) ** (1.0 / p)

    return AdditiveGenerator(
        name=f"Schweizer-Sklar(p={p:g})",
        family="Schweizer-Sklar",
        formula="f_p(x) = (1 - x**p) / p, p != 0; limit p=0 is -log(x)",
        f=f,
        finv=finv,
        f0_plus=f0_plus,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


def hamacher(p: float) -> AdditiveGenerator:
    if p < 0.0 or p == inf:
        raise ValueError("Hamacher additive generator requires 0 <= p < +inf.")
    if isclose(p, 0.0, abs_tol=EPS):
        return AdditiveGenerator(
            name="Hamacher(p=0)",
            family="Hamacher",
            formula="f_0(x) = (1 - x) / x, f(0)=+inf",
            f=lambda x: inf if x == 0.0 else (1.0 - x) / x,
            finv=lambda s: 1.0 / (1.0 + s),
            f0_plus=inf,
            parameter=0.0,
            reference_note="Hamacher product.",
            guarantee=BY_CONSTRUCTION,
        )

    def f(x: float) -> float:
        if x == 0.0:
            return inf
        return log1p(p * (1.0 - x) / x)

    def finv(s: float) -> float:
        if s == inf or s > MAX_EXP_ARGUMENT:
            return 0.0
        return p / (expm1(s) + p)

    return AdditiveGenerator(
        name=f"Hamacher(p={p:g})",
        family="Hamacher",
        formula="f_p(x) = log((p + (1-p)x) / x), p > 0",
        f=f,
        finv=finv,
        f0_plus=inf,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


def frank(p: float) -> AdditiveGenerator:
    if p <= 0.0:
        raise ValueError("Frank additive generator requires p > 0.")
    if isclose(p, 1.0, abs_tol=NEAR_LIMIT_EPS):
        base = product()
        return AdditiveGenerator(
            name="Frank(p=1)",
            family="Frank",
            formula=base.formula,
            f=base.f,
            finv=base.finv,
            f0_plus=base.f0_plus,
            parameter=1.0,
            reference_note="Limit case equal to the product t-norm.",
            guarantee=BY_CONSTRUCTION,
        )
    if p == inf:
        base = lukasiewicz()
        return AdditiveGenerator(
            name="Frank(p=+inf)",
            family="Frank",
            formula=base.formula,
            f=base.f,
            finv=base.finv,
            f0_plus=base.f0_plus,
            parameter=inf,
            reference_note="Limit case equal to the Lukasiewicz t-norm.",
            guarantee=BY_CONSTRUCTION,
        )

    log_p = log(p)
    numerator = expm1(log_p)

    def f(x: float) -> float:
        if x == 0.0:
            return inf
        denominator = expm1(x * log_p)
        return log(numerator / denominator)

    def finv(s: float) -> float:
        return log1p(numerator * exp(-s)) / log_p

    return AdditiveGenerator(
        name=f"Frank(p={p:g})",
        family="Frank",
        formula="f_p(x) = log((p - 1) / (p**x - 1)), p > 0, p != 1",
        f=f,
        finv=finv,
        f0_plus=inf,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


def yager(p: float) -> AdditiveGenerator:
    _positive(p)
    if p == inf:
        raise ValueError("Yager p=+inf is the minimum t-norm and has no additive generator.")
    return AdditiveGenerator(
        name=f"Yager(p={p:g})",
        family="Yager",
        formula="f_p(x) = (1 - x)**p",
        f=lambda x: (1.0 - x) ** p,
        finv=lambda s: max(0.0, 1.0 - s ** (1.0 / p)),
        f0_plus=1.0,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


def aczel_alsina(p: float) -> AdditiveGenerator:
    _positive(p)
    if p == inf:
        raise ValueError("Aczel-Alsina p=+inf is the minimum t-norm and has no additive generator.")

    def f(x: float) -> float:
        return inf if x == 0.0 else (-log(x)) ** p

    return AdditiveGenerator(
        name=f"Aczel-Alsina(p={p:g})",
        family="Aczel-Alsina",
        formula="f_p(x) = (-log(x))**p, f(0)=+inf",
        f=f,
        finv=lambda s: exp(-(s ** (1.0 / p))),
        f0_plus=inf,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


def dombi(p: float) -> AdditiveGenerator:
    _positive(p)
    if p == inf:
        raise ValueError("Dombi p=+inf is the minimum t-norm and has no additive generator.")

    def f(x: float) -> float:
        if x == 0.0:
            return inf
        return ((1.0 - x) / x) ** p

    return AdditiveGenerator(
        name=f"Dombi(p={p:g})",
        family="Dombi",
        formula="f_p(x) = ((1 - x) / x)**p, f(0)=+inf",
        f=f,
        finv=lambda s: 1.0 / (1.0 + s ** (1.0 / p)),
        f0_plus=inf,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


def sugeno_weber(p: float) -> AdditiveGenerator:
    if p <= -1.0:
        raise ValueError("Sugeno-Weber additive generator requires p > -1.")
    if p == inf:
        base = product()
        return AdditiveGenerator(
            name="Sugeno-Weber(p=+inf)",
            family="Sugeno-Weber",
            formula=base.formula,
            f=base.f,
            finv=base.finv,
            f0_plus=base.f0_plus,
            parameter=inf,
            reference_note="Limit case equal to the product t-norm.",
            guarantee=BY_CONSTRUCTION,
        )
    if isclose(p, 0.0, abs_tol=NEAR_LIMIT_EPS):
        base = lukasiewicz()
        return AdditiveGenerator(
            name="Sugeno-Weber(p=0)",
            family="Sugeno-Weber",
            formula=base.formula,
            f=base.f,
            finv=base.finv,
            f0_plus=base.f0_plus,
            parameter=0.0,
            reference_note="Case equal to the Lukasiewicz t-norm.",
            guarantee=BY_CONSTRUCTION,
        )

    log_base = log1p(p)

    def f(x: float) -> float:
        return 1.0 - log1p(p * x) / log_base

    def finv(s: float) -> float:
        if s >= 1.0:
            return 0.0
        return expm1((1.0 - s) * log_base) / p

    return AdditiveGenerator(
        name=f"Sugeno-Weber(p={p:g})",
        family="Sugeno-Weber",
        formula="f_p(x) = 1 - log(1 + p*x) / log(1 + p), p > -1, p != 0",
        f=f,
        finv=finv,
        f0_plus=1.0,
        parameter=p,
        guarantee=BY_CONSTRUCTION,
    )


KNOWN_GENERATOR_FACTORIES = {
    "lukasiewicz": lukasiewicz,
    "product": product,
    "drastic": drastic,
    "schweizer_sklar": schweizer_sklar,
    "hamacher": hamacher,
    "frank": frank,
    "yager": yager,
    "aczel_alsina": aczel_alsina,
    "dombi": dombi,
    "sugeno_weber": sugeno_weber,
}


def _sample_inverse_arguments(f_values: dict[float, float], f0_plus: float) -> list[float]:
    finite_values = sorted(
        {
            value
            for value in f_values.values()
            if isfinite(value) and value >= 0.0
        }
    )
    if f0_plus != inf and isfinite(f0_plus):
        finite_values.append(f0_plus)
    return sorted(set(finite_values))


def _validation_subset(points: list[float], *, limit: int) -> list[float]:
    if len(points) <= limit:
        return points
    indexes = {
        round(index * (len(points) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [points[index] for index in sorted(indexes)]


def known_literature_generators(sample_parameter: float = 2.0) -> list[AdditiveGenerator]:
    """Return one representative from each implemented literature family."""

    return [
        lukasiewicz(),
        product(),
        drastic(),
        schweizer_sklar(sample_parameter),
        hamacher(sample_parameter),
        frank(sample_parameter),
        yager(sample_parameter),
        aczel_alsina(sample_parameter),
        dombi(sample_parameter),
        sugeno_weber(sample_parameter),
    ]


def literature_summary() -> str:
    lines = [
        "Known implemented additive generators / families:",
        "- Lukasiewicz: f(x)=1-x",
        "- Product: f(x)=-log(x)",
        "- Drastic: f(1)=0, f(x)=2-x for x<1",
        "- Schweizer-Sklar: f_p(x)=(1-x**p)/p; p=0 gives product",
        "- Hamacher: f_0(x)=(1-x)/x; f_p(x)=log((p+(1-p)x)/x)",
        "- Frank: f_p(x)=log((p-1)/(p**x-1)); p=1 gives product; p=+inf gives Lukasiewicz",
        "- Yager: f_p(x)=(1-x)**p",
        "- Aczel-Alsina: f_p(x)=(-log(x))**p",
        "- Dombi: f_p(x)=((1-x)/x)**p",
        "- Sugeno-Weber: f_p(x)=1-log(1+p*x)/log(1+p); p=0 gives Lukasiewicz",
        "",
        "Boundary notes:",
        "- Minimum t-norm: no additive generator.",
        "- Positive multiples of a generator produce the same t-norm.",
        "- Ordinal sums build many continuous t-norms but usually not via one global additive generator.",
    ]
    return "\n".join(lines)
