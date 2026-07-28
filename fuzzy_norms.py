"""Basic fuzzy t-norms and t-conorms on [0,1]."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite
from typing import Callable

from tnorm_additive_generators import (
    aczel_alsina as aczel_alsina_generator,
    dombi as dombi_generator,
    frank as frank_generator,
    hamacher as hamacher_generator,
    schweizer_sklar as schweizer_sklar_generator,
    sugeno_weber as sugeno_weber_generator,
    yager as yager_generator,
)
from operator_guarantees import BY_CONSTRUCTION, normalize_guarantee


BinaryFunction = Callable[[float, float], float]
EPS = 1e-9


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
    if x < -EPS or x > 1.0 + EPS:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


@dataclass(frozen=True)
class FuzzyNormProperties:
    """Properties checked empirically on a finite grid."""

    maps_unit_square: bool
    empirically_commutative_on_grid: bool
    empirically_associative_on_grid: bool
    empirically_non_decreasing_on_grid: bool
    neutral_element: bool
    absorbing_boundary: bool
    valid: bool

    @property
    def commutative(self) -> bool:
        return self.empirically_commutative_on_grid

    @property
    def associative(self) -> bool:
        return self.empirically_associative_on_grid

    @property
    def non_decreasing(self) -> bool:
        return self.empirically_non_decreasing_on_grid


@dataclass(frozen=True)
class TNorm:
    """Triangular norm T:[0,1]^2->[0,1]."""

    name: str
    formula: str
    t: BinaryFunction
    family: str | None = None
    parameter: float | None = None
    reference_note: str = ""
    guarantee: str | None = None
    validate: bool = True
    validation_points: int = 41
    numerically_checked_properties: FuzzyNormProperties = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=self.validate),
        )
        properties = _check_binary_fuzzy_norm(
            self.t,
            kind="t_norm",
            validation_points=self.validation_points,
        )
        object.__setattr__(self, "numerically_checked_properties", properties)
        if self.validate and not properties.valid:
            raise ValueError(f"{self.name} is not a valid t-norm: {properties}.")

    def __call__(self, x: float, y: float) -> float:
        x = _unit_interval(x, "x")
        y = _unit_interval(y, "y")
        return _validate_unit_output(self.t(x, y), f"{self.name}({x}, {y})")

    @property
    def verified_properties(self) -> FuzzyNormProperties:
        """Backward-compatible alias for numerically_checked_properties."""

        return self.numerically_checked_properties

    def dual_t_conorm(self, *, name: str | None = None) -> "TConorm":
        """Return the standard dual t-conorm S(x,y)=1-T(1-x,1-y)."""

        return TConorm(
            name=name or f"Dual of {self.name}",
            formula=f"S(x,y)=1-{self.name}(1-x,1-y)",
            s=lambda x, y: 1.0 - self(1.0 - x, 1.0 - y),
            family=self.family,
            parameter=self.parameter,
            reference_note=f"Standard dual t-conorm of {self.name}.",
            guarantee=BY_CONSTRUCTION,
        )


@dataclass(frozen=True)
class TConorm:
    """Triangular conorm S:[0,1]^2->[0,1]."""

    name: str
    formula: str
    s: BinaryFunction
    family: str | None = None
    parameter: float | None = None
    reference_note: str = ""
    guarantee: str | None = None
    validate: bool = True
    validation_points: int = 41
    numerically_checked_properties: FuzzyNormProperties = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=self.validate),
        )
        properties = _check_binary_fuzzy_norm(
            self.s,
            kind="t_conorm",
            validation_points=self.validation_points,
        )
        object.__setattr__(self, "numerically_checked_properties", properties)
        if self.validate and not properties.valid:
            raise ValueError(f"{self.name} is not a valid t-conorm: {properties}.")

    def __call__(self, x: float, y: float) -> float:
        x = _unit_interval(x, "x")
        y = _unit_interval(y, "y")
        return _validate_unit_output(self.s(x, y), f"{self.name}({x}, {y})")

    @property
    def verified_properties(self) -> FuzzyNormProperties:
        """Backward-compatible alias for numerically_checked_properties."""

        return self.numerically_checked_properties

    def dual_t_norm(self, *, name: str | None = None) -> TNorm:
        """Return the standard dual t-norm T(x,y)=1-S(1-x,1-y)."""

        return TNorm(
            name=name or f"Dual of {self.name}",
            formula=f"T(x,y)=1-{self.name}(1-x,1-y)",
            t=lambda x, y: 1.0 - self(1.0 - x, 1.0 - y),
            family=self.family,
            parameter=self.parameter,
            reference_note=f"Standard dual t-norm of {self.name}.",
            guarantee=BY_CONSTRUCTION,
        )


def minimum() -> TNorm:
    return TNorm(
        name="Minimum",
        family="basic",
        formula="T_M(x,y)=min(x,y)",
        t=min,
        reference_note="Largest t-norm.",
        guarantee=BY_CONSTRUCTION,
    )


def maximum() -> TConorm:
    return TConorm(
        name="Maximum",
        family="basic",
        formula="S_M(x,y)=max(x,y)",
        s=max,
        reference_note="Smallest t-conorm; dual of minimum t-norm.",
        guarantee=BY_CONSTRUCTION,
    )


def product() -> TNorm:
    return TNorm(
        name="Product",
        family="basic",
        formula="T_P(x,y)=x*y",
        t=lambda x, y: x * y,
        guarantee=BY_CONSTRUCTION,
    )


def probabilistic_sum() -> TConorm:
    return TConorm(
        name="ProbabilisticSum",
        family="basic",
        formula="S_P(x,y)=x+y-x*y",
        s=lambda x, y: x + y - x * y,
        reference_note="Dual of product t-norm.",
        guarantee=BY_CONSTRUCTION,
    )


def lukasiewicz_t_norm() -> TNorm:
    return TNorm(
        name="Lukasiewicz t-norm",
        family="Lukasiewicz",
        formula="T_L(x,y)=max(0,x+y-1)",
        t=lambda x, y: max(0.0, x + y - 1.0),
        guarantee=BY_CONSTRUCTION,
    )


def lukasiewicz_t_conorm() -> TConorm:
    return TConorm(
        name="Lukasiewicz t-conorm",
        family="Lukasiewicz",
        formula="S_L(x,y)=min(1,x+y)",
        s=lambda x, y: min(1.0, x + y),
        guarantee=BY_CONSTRUCTION,
    )


def drastic_t_norm() -> TNorm:
    def t(x: float, y: float) -> float:
        if y == 1.0:
            return x
        if x == 1.0:
            return y
        return 0.0

    return TNorm(
        name="Drastic t-norm",
        family="Drastic",
        formula="T_D(x,y)=min(x,y) if max(x,y)=1; otherwise 0",
        t=t,
        guarantee=BY_CONSTRUCTION,
    )


def drastic_t_conorm() -> TConorm:
    def s(x: float, y: float) -> float:
        if y == 0.0:
            return x
        if x == 0.0:
            return y
        return 1.0

    return TConorm(
        name="Drastic t-conorm",
        family="Drastic",
        formula="S_D(x,y)=max(x,y) if min(x,y)=0; otherwise 1",
        s=s,
        guarantee=BY_CONSTRUCTION,
    )


def nilpotent_minimum() -> TNorm:
    def t(x: float, y: float) -> float:
        if x + y > 1.0:
            return min(x, y)
        return 0.0

    return TNorm(
        name="Nilpotent minimum",
        family="Nilpotent",
        formula="T_NM(x,y)=min(x,y) if x+y>1; otherwise 0",
        t=t,
        guarantee=BY_CONSTRUCTION,
    )


def nilpotent_maximum() -> TConorm:
    def s(x: float, y: float) -> float:
        if x + y < 1.0:
            return max(x, y)
        return 1.0

    return TConorm(
        name="Nilpotent maximum",
        family="Nilpotent",
        formula="S_NM(x,y)=max(x,y) if x+y<1; otherwise 1",
        s=s,
        guarantee=BY_CONSTRUCTION,
    )


def hamacher_product() -> TNorm:
    def t(x: float, y: float) -> float:
        denominator = x + y - x * y
        if denominator == 0.0:
            return 0.0
        return x * y / denominator

    return TNorm(
        name="Hamacher product",
        family="Hamacher",
        formula="T_H(x,y)=xy/(x+y-xy), with T_H(0,0)=0",
        t=t,
        guarantee=BY_CONSTRUCTION,
    )


def hamacher_sum() -> TConorm:
    def s(x: float, y: float) -> float:
        denominator = 1.0 - x * y
        if denominator == 0.0:
            return 1.0
        return (x + y - 2.0 * x * y) / denominator

    return TConorm(
        name="Hamacher sum",
        family="Hamacher",
        formula="S_H(x,y)=(x+y-2xy)/(1-xy), with S_H(1,1)=1",
        s=s,
        reference_note="Dual of Hamacher product.",
        guarantee=BY_CONSTRUCTION,
    )


def einstein_product() -> TNorm:
    return TNorm(
        name="Einstein product",
        family="Einstein",
        formula="T_E(x,y)=xy/(2-(x+y-xy))",
        t=lambda x, y: (x * y) / (2.0 - (x + y - x * y)),
        guarantee=BY_CONSTRUCTION,
    )


def einstein_sum() -> TConorm:
    return TConorm(
        name="Einstein sum",
        family="Einstein",
        formula="S_E(x,y)=(x+y)/(1+xy)",
        s=lambda x, y: (x + y) / (1.0 + x * y),
        reference_note="Dual of Einstein product.",
        guarantee=BY_CONSTRUCTION,
    )


def bounded_difference() -> TNorm:
    return lukasiewicz_t_norm()


def bounded_sum() -> TConorm:
    return lukasiewicz_t_conorm()


def t_norm_from_additive_generator(
    generator_factory: Callable[..., object],
    *args,
    name: str | None = None,
    **kwargs,
) -> TNorm:
    generator = generator_factory(*args, **kwargs)
    return TNorm(
        name=name or generator.name,
        formula=f"T(x,y) from additive generator {generator.formula}",
        t=generator.t_norm,
        family=generator.family,
        parameter=generator.parameter,
        reference_note=generator.reference_note,
        guarantee=BY_CONSTRUCTION,
    )


def schweizer_sklar(p: float) -> TNorm:
    return t_norm_from_additive_generator(schweizer_sklar_generator, p)


def schweizer_sklar_t_conorm(p: float) -> TConorm:
    return schweizer_sklar(p).dual_t_conorm(name=f"Schweizer-Sklar t-conorm(p={p:g})")


def yager(p: float) -> TNorm:
    return t_norm_from_additive_generator(yager_generator, p)


def yager_t_conorm(p: float) -> TConorm:
    return yager(p).dual_t_conorm(name=f"Yager t-conorm(p={p:g})")


def aczel_alsina(p: float) -> TNorm:
    return t_norm_from_additive_generator(aczel_alsina_generator, p)


def aczel_alsina_t_conorm(p: float) -> TConorm:
    return aczel_alsina(p).dual_t_conorm(name=f"Aczel-Alsina t-conorm(p={p:g})")


def dombi(p: float) -> TNorm:
    return t_norm_from_additive_generator(dombi_generator, p)


def dombi_t_conorm(p: float) -> TConorm:
    return dombi(p).dual_t_conorm(name=f"Dombi t-conorm(p={p:g})")


def frank(p: float) -> TNorm:
    return t_norm_from_additive_generator(frank_generator, p)


def frank_t_conorm(p: float) -> TConorm:
    return frank(p).dual_t_conorm(name=f"Frank t-conorm(p={p:g})")


def hamacher(p: float) -> TNorm:
    return t_norm_from_additive_generator(hamacher_generator, p)


def hamacher_t_conorm(p: float) -> TConorm:
    return hamacher(p).dual_t_conorm(name=f"Hamacher t-conorm(p={p:g})")


def sugeno_weber(p: float) -> TNorm:
    return t_norm_from_additive_generator(sugeno_weber_generator, p)


def sugeno_weber_t_conorm(p: float) -> TConorm:
    return sugeno_weber(p).dual_t_conorm(name=f"Sugeno-Weber t-conorm(p={p:g})")


def known_t_norms(sample_parameter: float = 2.0) -> list[TNorm]:
    """Return representative basic t-norms and parametric families."""

    return [
        minimum(),
        product(),
        lukasiewicz_t_norm(),
        drastic_t_norm(),
        nilpotent_minimum(),
        hamacher_product(),
        einstein_product(),
        schweizer_sklar(sample_parameter),
        yager(sample_parameter),
        aczel_alsina(sample_parameter),
        dombi(sample_parameter),
        frank(sample_parameter),
        hamacher(sample_parameter),
        sugeno_weber(sample_parameter),
    ]


def known_t_conorms(sample_parameter: float = 2.0) -> list[TConorm]:
    """Return representative basic t-conorms and parametric families."""

    return [
        maximum(),
        probabilistic_sum(),
        lukasiewicz_t_conorm(),
        drastic_t_conorm(),
        nilpotent_maximum(),
        hamacher_sum(),
        einstein_sum(),
        schweizer_sklar_t_conorm(sample_parameter),
        yager_t_conorm(sample_parameter),
        aczel_alsina_t_conorm(sample_parameter),
        dombi_t_conorm(sample_parameter),
        frank_t_conorm(sample_parameter),
        hamacher_t_conorm(sample_parameter),
        sugeno_weber_t_conorm(sample_parameter),
    ]


def fuzzy_norms_summary() -> str:
    lines = [
        "Known implemented fuzzy t-norms and t-conorms:",
        "- Minimum / maximum",
        "- Product / probabilistic sum",
        "- Lukasiewicz bounded difference / bounded sum",
        "- Drastic t-norm / drastic t-conorm",
        "- Nilpotent minimum / nilpotent maximum",
        "- Hamacher product / Hamacher sum",
        "- Einstein product / Einstein sum",
        "- Schweizer-Sklar, Yager, Aczel-Alsina, Dombi, Frank, Hamacher, Sugeno-Weber families",
    ]
    return "\n".join(lines)


def _check_binary_fuzzy_norm(
    function: BinaryFunction,
    *,
    kind: str,
    validation_points: int,
) -> FuzzyNormProperties:
    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    points = [
        index / (validation_points - 1)
        for index in range(validation_points)
    ]
    axiom_points = _validation_subset(points, limit=13)

    maps_unit_square = True
    values: dict[tuple[float, float], float] = {}
    for x in axiom_points:
        for y in axiom_points:
            try:
                values[(x, y)] = _validate_unit_output(function(x, y), f"F({x},{y})")
            except (ArithmeticError, ValueError, OverflowError):
                maps_unit_square = False
                values[(x, y)] = float("nan")

    commutative = maps_unit_square
    for x in axiom_points:
        for y in axiom_points:
            if abs(values[(x, y)] - values[(y, x)]) > 1e-6:
                commutative = False
                break

    non_decreasing = maps_unit_square
    for x1 in axiom_points:
        for x2 in axiom_points:
            if x1 > x2:
                continue
            for y1 in axiom_points:
                for y2 in axiom_points:
                    if y1 > y2:
                        continue
                    if values[(x1, y1)] > values[(x2, y2)] + 1e-6:
                        non_decreasing = False
                        break

    associativity_points = _validation_subset(points, limit=9)
    associative = maps_unit_square
    for x in associativity_points:
        for y in associativity_points:
            for z in associativity_points:
                left = function(function(x, y), z)
                right = function(x, function(y, z))
                if abs(left - right) > 1e-6:
                    associative = False
                    break

    neutral = 1.0 if kind == "t_norm" else 0.0
    absorbing = 0.0 if kind == "t_norm" else 1.0
    neutral_element = maps_unit_square
    absorbing_boundary = maps_unit_square
    for x in axiom_points:
        if (
            abs(function(x, neutral) - x) > 1e-6
            or abs(function(neutral, x) - x) > 1e-6
        ):
            neutral_element = False
            break
        if (
            abs(function(x, absorbing) - absorbing) > 1e-6
            or abs(function(absorbing, x) - absorbing) > 1e-6
        ):
            absorbing_boundary = False
            break

    valid = (
        maps_unit_square
        and commutative
        and associative
        and non_decreasing
        and neutral_element
        and absorbing_boundary
    )
    return FuzzyNormProperties(
        maps_unit_square=maps_unit_square,
        empirically_commutative_on_grid=commutative,
        empirically_associative_on_grid=associative,
        empirically_non_decreasing_on_grid=non_decreasing,
        neutral_element=neutral_element,
        absorbing_boundary=absorbing_boundary,
        valid=valid,
    )


def _validate_unit_output(value: float, name: str) -> float:
    return _clamp_unit(value, name)


def _validation_subset(points: list[float], *, limit: int) -> list[float]:
    if len(points) <= limit:
        return points
    indexes = {
        round(index * (len(points) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [points[index] for index in sorted(indexes)]


KNOWN_T_NORM_FACTORIES = {
    "minimum": minimum,
    "product": product,
    "lukasiewicz": lukasiewicz_t_norm,
    "drastic": drastic_t_norm,
    "nilpotent_minimum": nilpotent_minimum,
    "hamacher_product": hamacher_product,
    "einstein_product": einstein_product,
    "schweizer_sklar": schweizer_sklar,
    "yager": yager,
    "aczel_alsina": aczel_alsina,
    "dombi": dombi,
    "frank": frank,
    "hamacher": hamacher,
    "sugeno_weber": sugeno_weber,
}

KNOWN_T_CONORM_FACTORIES = {
    "maximum": maximum,
    "probabilistic_sum": probabilistic_sum,
    "lukasiewicz": lukasiewicz_t_conorm,
    "drastic": drastic_t_conorm,
    "nilpotent_maximum": nilpotent_maximum,
    "hamacher_sum": hamacher_sum,
    "einstein_sum": einstein_sum,
    "schweizer_sklar": schweizer_sklar_t_conorm,
    "yager": yager_t_conorm,
    "aczel_alsina": aczel_alsina_t_conorm,
    "dombi": dombi_t_conorm,
    "frank": frank_t_conorm,
    "hamacher": hamacher_t_conorm,
    "sugeno_weber": sugeno_weber_t_conorm,
}
