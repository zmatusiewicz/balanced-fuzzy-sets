"""Nullnorms on [0,1] and their transport to the balanced interval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite

from .operator_guarantees import BY_CONSTRUCTION, attach_guarantee, normalize_guarantee

NumberFunction = Callable[[float], float]
BinaryFunction = Callable[[float, float], float]
EPS = 1e-9


def _unit_interval(x: float, name: str = "x") -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}")
    return x


def _balanced_interval(x: float, name: str = "x") -> float:
    if not -1.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1], got {x!r}")
    return x


def _zero_element(z: float) -> float:
    if not 0.0 <= z <= 1.0:
        raise ValueError(f"z must be in [0, 1], got {z!r}")
    return z


def _finite(value: float, name: str) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _validate_unit_output(value: float, name: str) -> float:
    _finite(value, name)
    if value < -EPS or value > 1.0 + EPS:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    if abs(value) <= EPS:
        return 0.0
    if abs(value - 1.0) <= EPS:
        return 1.0
    return value


def _validate_balanced_output(value: float, name: str) -> float:
    _finite(value, name)
    if value < -1.0 - EPS or value > 1.0 + EPS:
        raise ValueError(f"{name} must be in [-1, 1], got {value!r}.")
    if abs(value) <= EPS:
        return 0.0
    if abs(value - 1.0) <= EPS:
        return 1.0
    if abs(value + 1.0) <= EPS:
        return -1.0
    return value


@dataclass(frozen=True)
class NullnormProperties:
    """Properties checked empirically on a finite grid."""

    maps_unit_interval: bool
    empirically_commutative_on_grid: bool
    empirically_associative_on_grid: bool
    empirically_non_decreasing_on_grid: bool
    has_zero_element: bool
    has_boundary_neutrality: bool
    valid_nullnorm: bool

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
class Nullnorm:
    """Nullnorm V:[0,1]^2->[0,1] with zero element z."""

    name: str
    formula: str
    z: float
    v: BinaryFunction
    reference_note: str = ""
    guarantee: str | None = None
    validate: bool = True
    validation_points: int = 41
    numerically_checked_properties: NullnormProperties = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guarantee",
            normalize_guarantee(self.guarantee, validate=self.validate),
        )
        _zero_element(self.z)
        properties = self.check_properties()
        object.__setattr__(self, "numerically_checked_properties", properties)
        if self.validate and not properties.valid_nullnorm:
            failed = [
                name
                for name in (
                    "maps_unit_interval",
                    "empirically_commutative_on_grid",
                    "empirically_associative_on_grid",
                    "empirically_non_decreasing_on_grid",
                    "has_zero_element",
                    "has_boundary_neutrality",
                )
                if not getattr(properties, name)
            ]
            raise ValueError(
                f"{self.name} is not a valid nullnorm; failed properties: "
                f"{', '.join(failed)}."
            )

    def __call__(self, x: float, y: float) -> float:
        x = _unit_interval(x, "x")
        y = _unit_interval(y, "y")
        return _validate_unit_output(self.v(x, y), f"{self.name}({x}, {y})")

    @property
    def verified_properties(self) -> NullnormProperties:
        """Backward-compatible alias for numerically_checked_properties."""

        return self.numerically_checked_properties

    def check_properties(self) -> NullnormProperties:
        """Check nullnorm axioms on a finite grid."""

        if self.validation_points < 3:
            raise ValueError("validation_points must be at least 3.")

        points = [
            index / (self.validation_points - 1)
            for index in range(self.validation_points)
        ]

        maps_unit_interval = True
        values: dict[tuple[float, float], float] = {}
        for x in points:
            for y in points:
                try:
                    values[(x, y)] = _validate_unit_output(
                        self.v(x, y),
                        f"{self.name}({x}, {y})",
                    )
                except ValueError:
                    maps_unit_interval = False
                    values[(x, y)] = float("nan")

        commutative = maps_unit_interval
        for x in points:
            for y in points:
                if abs(values[(x, y)] - values[(y, x)]) > 1e-6:
                    commutative = False
                    break

        non_decreasing = maps_unit_interval
        for x1 in points:
            for x2 in points:
                if x1 > x2:
                    continue
                for y1 in points:
                    for y2 in points:
                        if y1 > y2:
                            continue
                        if values[(x1, y1)] > values[(x2, y2)] + 1e-6:
                            non_decreasing = False
                            break

        has_zero_element = maps_unit_interval
        for x in points:
            if (
                abs(_validate_unit_output(self.v(self.z, x), f"{self.name}(z,{x})") - self.z)
                > 1e-6
                or abs(_validate_unit_output(self.v(x, self.z), f"{self.name}({x},z)") - self.z)
                > 1e-6
            ):
                has_zero_element = False
                break

        has_boundary_neutrality = maps_unit_interval
        for x in points:
            if x <= self.z + EPS:
                left = _validate_unit_output(self.v(x, 0.0), f"{self.name}({x},0)")
                right = _validate_unit_output(self.v(0.0, x), f"{self.name}(0,{x})")
                if abs(left - x) > 1e-6 or abs(right - x) > 1e-6:
                    has_boundary_neutrality = False
                    break
            if x >= self.z - EPS:
                left = _validate_unit_output(self.v(x, 1.0), f"{self.name}({x},1)")
                right = _validate_unit_output(self.v(1.0, x), f"{self.name}(1,{x})")
                if abs(left - x) > 1e-6 or abs(right - x) > 1e-6:
                    has_boundary_neutrality = False
                    break

        associativity_points = _validation_subset(points, limit=11)
        associative = maps_unit_interval
        for x in associativity_points:
            for y in associativity_points:
                for z in associativity_points:
                    left = self.v(self.v(x, y), z)
                    right = self.v(x, self.v(y, z))
                    if abs(left - right) > 1e-6:
                        associative = False
                        break

        valid_nullnorm = (
            maps_unit_interval
            and commutative
            and associative
            and non_decreasing
            and has_zero_element
            and has_boundary_neutrality
        )

        return NullnormProperties(
            maps_unit_interval=maps_unit_interval,
            empirically_commutative_on_grid=commutative,
            empirically_associative_on_grid=associative,
            empirically_non_decreasing_on_grid=non_decreasing,
            has_zero_element=has_zero_element,
            has_boundary_neutrality=has_boundary_neutrality,
            valid_nullnorm=valid_nullnorm,
        )


def nullnorm_from_t_norm_and_t_conorm(
    t_norm: BinaryFunction,
    t_conorm: BinaryFunction | None = None,
    *,
    z: float,
    name: str | None = None,
    validate: bool = True,
) -> Nullnorm:
    """
    Build a nullnorm from a t-conorm below z and a t-norm above z.

    V(x,y)=z*S(x/z,y/z) for x,y<=z,
    V(x,y)=z for min(x,y)<=z<=max(x,y),
    V(x,y)=z+(1-z)*T((x-z)/(1-z),(y-z)/(1-z)) for x,y>=z.
    """

    z = _zero_element(z)

    if t_conorm is None:

        def t_conorm(x: float, y: float) -> float:
            x = _unit_interval(x, "x")
            y = _unit_interval(y, "y")
            return 1.0 - t_norm(1.0 - x, 1.0 - y)

    def v(x: float, y: float) -> float:
        x = _unit_interval(x, "x")
        y = _unit_interval(y, "y")
        if x <= z and y <= z:
            if z == 0.0:
                return 0.0
            return z * t_conorm(x / z, y / z)
        if x >= z and y >= z:
            if z == 1.0:
                return 1.0
            return z + (1.0 - z) * t_norm(
                (x - z) / (1.0 - z),
                (y - z) / (1.0 - z),
            )
        return z

    return Nullnorm(
        name=name or f"Nullnorm(z={z:g})",
        formula=(
            "piecewise nullnorm generated by a t-conorm below z and "
            "a t-norm above z"
        ),
        z=z,
        v=v,
        guarantee=BY_CONSTRUCTION,
        validate=validate,
    )


def transport_nullnorm(
    nullnorm: Nullnorm,
    f: NumberFunction,
    f_inverse: NumberFunction,
    *,
    name: str | None = None,
    validate: bool = True,
    validation_points: int = 401,
) -> BinaryFunction:
    """
    Transport a nullnorm V on [0,1] to [-1,1] by isomorphism.

    This is the general transported operation. It validates the transport map
    and the transported zero element, but it does not claim balanced symmetry.
    """

    if validate:
        _validate_transport(nullnorm, f, f_inverse, validation_points)

    def transported(x: float, y: float) -> float:
        x = _balanced_interval(x, "x")
        y = _balanced_interval(y, "y")
        result = f_inverse(nullnorm(f(x), f(y)))
        return _validate_balanced_output(result, name or "transported nullnorm")

    return attach_guarantee(transported, BY_CONSTRUCTION, validate=validate)


def balanced_t_norm_from_nullnorm(
    nullnorm: Nullnorm,
    f: NumberFunction,
    f_inverse: NumberFunction,
    *,
    name: str | None = None,
    validate: bool = True,
    validation_points: int = 401,
) -> BinaryFunction:
    """
    Construct a balanced t-norm from a nullnorm under theorem-level assumptions.

    In addition to transport validity, the result must satisfy the balanced
    symmetry P(x,y)=-P(-x,-y).
    """

    operation = transport_nullnorm(
        nullnorm,
        f,
        f_inverse,
        name=name or "balanced t-norm from nullnorm",
        validate=validate,
        validation_points=validation_points,
    )
    if validate:
        _validate_balanced_symmetry(operation, validation_points)
    return operation


def transport_nullnorm_to_balanced_interval(
    nullnorm: Nullnorm,
    f: NumberFunction,
    f_inverse: NumberFunction,
    *,
    name: str | None = None,
    validate: bool = True,
    validation_points: int = 401,
) -> BinaryFunction:
    """Backward-compatible alias for transport_nullnorm."""

    return transport_nullnorm(
        nullnorm,
        f,
        f_inverse,
        name=name,
        validate=validate,
        validation_points=validation_points,
    )


def _validate_transport(
    nullnorm: Nullnorm,
    f: NumberFunction,
    f_inverse: NumberFunction,
    validation_points: int,
) -> None:
    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    checks = [
        ("f(-1)", f(-1.0), 0.0),
        ("f(0)", f(0.0), nullnorm.z),
        ("f(1)", f(1.0), 1.0),
        ("f_inverse(0)", f_inverse(0.0), -1.0),
        ("f_inverse(z)", f_inverse(nullnorm.z), 0.0),
        ("f_inverse(1)", f_inverse(1.0), 1.0),
    ]
    for label, actual, expected in checks:
        _finite(actual, label)
        if abs(actual - expected) > EPS:
            raise ValueError(f"{label} must be {expected!r}, got {actual!r}.")

    xs = [
        -1.0 + 2.0 * index / (validation_points - 1)
        for index in range(validation_points)
    ]
    us = [
        index / (validation_points - 1)
        for index in range(validation_points)
    ]

    previous = None
    for x in xs:
        fx = _validate_unit_output(f(x), f"f({x})")
        if previous is not None and fx <= previous + EPS:
            raise ValueError("f must be strictly increasing on [-1,1].")
        previous = fx
        roundtrip = _validate_balanced_output(f_inverse(fx), f"f_inverse(f({x}))")
        if abs(roundtrip - x) > 1e-6:
            raise ValueError(
                f"f_inverse must invert f; f_inverse(f({x}))={roundtrip!r}."
            )

    previous = None
    for u in us:
        x = _validate_balanced_output(f_inverse(u), f"f_inverse({u})")
        if previous is not None and x <= previous + EPS:
            raise ValueError("f_inverse must be strictly increasing on [0,1].")
        previous = x
        roundtrip = _validate_unit_output(f(x), f"f(f_inverse({u}))")
        if abs(roundtrip - u) > 1e-6:
            raise ValueError(
                f"f must invert f_inverse; f(f_inverse({u}))={roundtrip!r}."
            )

    transported_zero = f_inverse(nullnorm.z)
    if abs(transported_zero) > EPS:
        raise ValueError(f"Transported zero element must be 0, got {transported_zero!r}.")

    for x in xs:
        fx = f(x)
        left = f_inverse(nullnorm(fx, nullnorm.z))
        right = f_inverse(nullnorm(nullnorm.z, fx))
        if abs(left) > 1e-6 or abs(right) > 1e-6:
            raise ValueError(
                "Transported nullnorm must have zero element 0; "
                f"got V_B({x},0)={left!r}, V_B(0,{x})={right!r}."
            )


def _validate_balanced_symmetry(
    operation: BinaryFunction,
    validation_points: int,
) -> None:
    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    points = [
        -1.0 + 2.0 * index / (validation_points - 1)
        for index in range(validation_points)
    ]
    for x in points:
        for y in points:
            value = _validate_balanced_output(operation(x, y), f"P({x},{y})")
            opposite = _validate_balanced_output(operation(-x, -y), f"P({-x},{-y})")
            if abs(value + opposite) > 1e-6:
                raise ValueError(
                    "balanced symmetry failed: "
                    f"P({x},{y})={value!r}, -P({-x},{-y})={-opposite!r}."
                )


def _validation_subset(points: list[float], *, limit: int) -> list[float]:
    if len(points) <= limit:
        return points
    indexes = {
        round(index * (len(points) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [points[index] for index in sorted(indexes)]


def nullnorms_summary() -> str:
    return (
        "Implemented nullnorm support:\n"
        "- Nullnorm type V:[0,1]^2->[0,1] with zero element z\n"
        "- Piecewise constructor from t-conorm below z and t-norm above z\n"
        "- Numerical validation of commutativity, associativity, monotonicity, "
        "z, and boundary neutralities\n"
        "- General transport to [-1,1] with f(0)=z and transported zero element "
        "equal to 0\n"
        "- Balanced t-norm construction from nullnorm with additional symmetry "
        "validation"
    )
