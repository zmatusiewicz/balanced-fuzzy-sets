"""Factories for balanced fuzzy t-norms and t-conorms."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, isinf

from .balanced_fuzzy_negations import BalancedFuzzyNegation
from .fuzzy_negations import FuzzyNegation
from .nullnorms import (
    Nullnorm,
)
from .nullnorms import (
    balanced_t_norm_from_nullnorm as _balanced_t_norm_from_nullnorm,
)
from .nullnorms import (
    transport_nullnorm as _transport_nullnorm,
)
from .operator_guarantees import BY_CONSTRUCTION, attach_guarantee
from .representable_uninorm_generator import AdditiveGeneratorOfRepresentableUninorm

NumberFunction = Callable[[float], float]
BinaryFunction = Callable[[float, float], float]
EPS = 1e-9


def _unit_interval(x: float, name: str) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}")
    return x


def _balanced_interval(x: float, name: str) -> float:
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


class BalancedFuzzyOperators:
    """Factories for balanced fuzzy operations."""

    @staticmethod
    def create_balanced_negation_from_function(
        function: NumberFunction,
        *,
        name: str = "Generated balanced fuzzy negation",
        formula: str = "N_B(x)",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
        is_continuous: bool | None = None,
        reference_note: str = "",
        guarantee: str | None = None,
    ) -> BalancedFuzzyNegation:
        """Wrap a decreasing function N_B:[-1,1]->[-1,1] with N_B(0)=0."""

        if abs(function(0.0)) > 1e-12:
            raise ValueError("Balanced fuzzy negation must satisfy N_B(0)=0.")

        def generated(x: float) -> float:
            # Formula: N_B(x)=function(x), where function(0)=0 and
            # function maps [-1,1] into [-1,1].
            return function(_balanced_interval(x, "x"))

        return BalancedFuzzyNegation(
            name=name,
            formula=formula,
            n=generated,
            is_strict=is_strict,
            is_strong=is_strong,
            is_continuous=is_continuous,
            reference_note=reference_note,
            guarantee=guarantee,
        )

    @staticmethod
    def create_balanced_negation_from_one_fuzzy_negation(
        negation: NumberFunction | FuzzyNegation,
        *,
        name: str | None = None,
        formula: str | None = None,
        is_strict: bool | None = None,
        is_strong: bool | None = None,
        is_continuous: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """
        Generate BFN from one fuzzy negation n.

        N_B(x)=n(x)-1 for x>0, N_B(x)=1-n(-x) for x<0, and N_B(0)=0.
        """

        n = negation if callable(negation) else negation.n
        source_name = getattr(negation, "name", "fuzzy negation")
        source_formula = getattr(negation, "formula", "n(x)")

        def generated(x: float) -> float:
            # Formula:
            #   N_B(x)=n(x)-1    for x in (0,1],
            #   N_B(x)=1-n(-x)   for x in [-1,0),
            #   N_B(0)=0.
            x = _balanced_interval(x, "x")
            if x > 0.0:
                return _unit_interval_with_tolerance(n(x), "n(x)") - 1.0
            if x < 0.0:
                return 1.0 - _unit_interval_with_tolerance(n(-x), "n(-x)")
            return 0.0

        return BalancedFuzzyNegation(
            name=name or f"Balanced from {source_name}",
            formula=formula or f"N_B(x)={source_formula}-1 for x>0; 1-n(-x) for x<0",
            n=generated,
            is_strict=is_strict,
            is_strong=is_strong,
            is_continuous=is_continuous,
            reference_note="Generated from one fuzzy negation on [0,1].",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_two_fuzzy_negations(
        positive_negation: NumberFunction | FuzzyNegation,
        negative_negation: NumberFunction | FuzzyNegation,
        *,
        name: str = "Balanced from two fuzzy negations",
        formula: str = "N_B(x)=n1(x)-1 for x>0; N_B(x)=1-n2(-x) for x<0",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN with different fuzzy negations on positive and negative sides."""

        n1 = positive_negation if callable(positive_negation) else positive_negation.n
        n2 = negative_negation if callable(negative_negation) else negative_negation.n

        def generated(x: float) -> float:
            # Formula:
            #   N_B(x)=n1(x)-1    for x in (0,1],
            #   N_B(x)=1-n2(-x)   for x in [-1,0),
            #   N_B(0)=0.
            x = _balanced_interval(x, "x")
            if x > 0.0:
                return _unit_interval_with_tolerance(n1(x), "n1(x)") - 1.0
            if x < 0.0:
                return 1.0 - _unit_interval_with_tolerance(n2(-x), "n2(-x)")
            return 0.0

        return BalancedFuzzyNegation(
            name=name,
            formula=formula,
            n=generated,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated with independent fuzzy negations on both sides.",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_bijection(
        function: NumberFunction,
        *,
        name: str = "Balanced from bijection",
        formula: str = "N_B(x)=f(x)",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN from a bijection f with f(1)=-1, f(0)=0, f(-1)=1."""

        # Formula: N_B(x)=f(x), with f(1)=-1, f(0)=0, f(-1)=1.
        if abs(function(1.0) + 1.0) > 1e-12:
            raise ValueError("function must satisfy f(1)=-1.")
        if abs(function(0.0)) > 1e-12:
            raise ValueError("function must satisfy f(0)=0.")
        if abs(function(-1.0) - 1.0) > 1e-12:
            raise ValueError("function must satisfy f(-1)=1.")

        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            function,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated from a normalized bijection on [-1,1].",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_scaled_bijection(
        function: NumberFunction,
        scale: float,
        *,
        name: str = "Balanced from scaled bijection",
        formula: str = "N_B(x)=f(x)/lambda",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN from f with f(1)=-lambda, f(0)=0, f(-1)=lambda."""

        # Formula: N_B(x)=f(x)/lambda, with
        # f(1)=-lambda, f(0)=0, f(-1)=lambda, lambda != 0.
        if scale == 0.0:
            raise ValueError("scale must be non-zero.")
        if abs(function(1.0) + scale) > 1e-12:
            raise ValueError("function must satisfy f(1)=-scale.")
        if abs(function(0.0)) > 1e-12:
            raise ValueError("function must satisfy f(0)=0.")
        if abs(function(-1.0) - scale) > 1e-12:
            raise ValueError("function must satisfy f(-1)=scale.")

        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            lambda x: function(x) / scale,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated from a scaled bijection on [-1,1].",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_inverse_balanced_negation(
        inverse_function: NumberFunction,
        *,
        name: str = "Inverse balanced fuzzy negation",
        formula: str = "N_B^{-1}(x)",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Create BFN from the inverse function of an existing BFN."""

        # Formula: if N_B is a BFN and inverse_function=N_B^{-1},
        # then inverse_function is also a BFN.
        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            inverse_function,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated as inverse of a balanced fuzzy negation.",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_odd_composition_balanced_negation(
        negations: list[BalancedFuzzyNegation | NumberFunction],
        *,
        name: str = "Odd composition balanced fuzzy negation",
        formula: str = "N_B=N_1 o ... o N_k, k odd",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Compose an odd number of BFNs; the result is again a BFN."""

        if len(negations) == 0 or len(negations) % 2 == 0:
            raise ValueError("negations must contain an odd positive number of functions.")

        def generated(x: float) -> float:
            # Formula: N_B=N_1 o N_2 o ... o N_k, where k is odd.
            result = _balanced_interval(x, "x")
            for negation in reversed(negations):
                result = negation(result)
                result = _balanced_interval(result, "composed result")
            return result

        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            generated,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated as composition of an odd number of balanced fuzzy negations.",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_even_extension(
        even_negation: NumberFunction,
        *,
        name: str = "Balanced from even fuzzy negation extension",
        formula: str = "N_B(x)=n(x)-1 for x>=0; N_B(x)=1-n(x) for x<0",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN from an even extension n:[-1,1]->[0,1]."""

        def generated(x: float) -> float:
            # Formula for even n:
            #   N_B(x)=n(x)-1   for x>=0,
            #   N_B(x)=1-n(x)   for x<0.
            x = _balanced_interval(x, "x")
            value = _unit_interval_with_tolerance(even_negation(x), "n(x)")
            if x >= 0.0:
                return value - 1.0
            return 1.0 - value

        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            generated,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated from an even extension of a fuzzy negation.",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_odd_generator(
        odd_generator: NumberFunction,
        *,
        name: str = "Balanced from odd generator",
        formula: str = "N_B(x)=-f(x)",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN from odd f where the fuzzy negation has form n(x)=1-f(x)."""

        # Formula for odd f: N_B(x)=-f(x).
        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            lambda x: -odd_generator(x),
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated from an odd function f with n(x)=1-f(x).",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_even_generator(
        even_generator: NumberFunction,
        *,
        name: str = "Balanced from even generator",
        formula: str = "N_B(x)=-f(x) for x>=0; N_B(x)=f(x) for x<0",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN from even f where the fuzzy negation has form n(x)=1-f(x)."""

        def generated(x: float) -> float:
            # Formula for even f:
            #   N_B(x)=-f(x)   for x>=0,
            #   N_B(x)= f(x)   for x<0.
            x = _balanced_interval(x, "x")
            value = _unit_interval_with_tolerance(even_generator(x), "f(x)")
            if x >= 0.0:
                return -value
            return value

        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            generated,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated from an even function f with n(x)=1-f(x).",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_negation_from_positive_generator(
        positive_generator: NumberFunction,
        *,
        name: str = "Balanced from positive-side generator",
        formula: str = "N_B(x)=-f(x) for x>=0; N_B(x)=f(-x) for x<0",
        is_strict: bool | None = None,
        is_strong: bool | None = None,
    ) -> BalancedFuzzyNegation:
        """Generate BFN from f defined for x>=0, with n(x)=1-f(x)."""

        def generated(x: float) -> float:
            # Formula for f defined on [0,1]:
            #   N_B(x)=-f(x)    for x>=0,
            #   N_B(x)= f(-x)   for x<0.
            x = _balanced_interval(x, "x")
            if x >= 0.0:
                value = _unit_interval_with_tolerance(positive_generator(x), "f(x)")
                return -value
            value = _unit_interval_with_tolerance(positive_generator(-x), "f(-x)")
            return value

        return BalancedFuzzyOperators.create_balanced_negation_from_function(
            generated,
            name=name,
            formula=formula,
            is_strict=is_strict,
            is_strong=is_strong,
            reference_note="Generated from a function defined on the non-negative side.",
            guarantee=BY_CONSTRUCTION,
        )

    @staticmethod
    def create_balanced_t_conorm(
        uninorm_generator: AdditiveGeneratorOfRepresentableUninorm,
        f: NumberFunction,
        f_inverse: NumberFunction,
        *,
        opposite_corner_convention: str = "disjunctive",
        balanced_negation: NumberFunction | BalancedFuzzyNegation | None = None,
        unit_negation: NumberFunction | FuzzyNegation | None = None,
        validate: bool = True,
        validation_points: int = 401,
    ) -> BinaryFunction:
        """
        Create balanced t-conorm S_B by transporting a representable uninorm.

        The operation is S_B(x,y)=f^{-1}(U(f(x),f(y))). At opposite corners
        (-1,1) and (1,-1), opposite_corner_convention chooses either
        "disjunctive" -> 1 or "conjunctive" -> -1.
        """

        if opposite_corner_convention == "disjunctive":
            opposite_corner_value = 1.0
        elif opposite_corner_convention == "conjunctive":
            opposite_corner_value = -1.0
        else:
            raise ValueError(
                "opposite_corner_convention must be 'disjunctive' or 'conjunctive'."
            )
        numerically_checked_properties = {
            "empirically_transport_valid_on_grid": False,
            "empirically_uninorm_generator_antisymmetric_on_grid": False,
            "empirically_symmetric_on_grid": False,
        }
        if validate:
            numerically_checked_properties.update(
                _validate_transport_to_balanced_interval(
                    uninorm_generator,
                    f,
                    f_inverse,
                    balanced_negation=balanced_negation,
                    unit_negation=unit_negation,
                    validation_points=validation_points,
                )
            )

        def balanced_t_conorm(x: float, y: float) -> float:
            x = _balanced_interval(x, "x")
            y = _balanced_interval(y, "y")

            if (x, y) in {(-1.0, 1.0), (1.0, -1.0)}:
                return opposite_corner_value

            ux = f(x)
            uy = f(y)
            ux = _unit_interval_with_tolerance(ux, "f(x)")
            uy = _unit_interval_with_tolerance(uy, "f(y)")

            result = f_inverse(
                uninorm_generator.uninorm(
                    ux,
                    uy,
                    infinity_convention=opposite_corner_convention,
                )
            )
            return _clamp_balanced(result, "S_B(x,y)")

        balanced_t_conorm.numerically_checked_properties = numerically_checked_properties
        balanced_t_conorm.verified_properties = numerically_checked_properties
        return attach_guarantee(balanced_t_conorm, BY_CONSTRUCTION, validate=validate)

    @staticmethod
    def balanced_t_norm_from_t_norm(
        t_norm: BinaryFunction,
        *,
        validate: bool = True,
        validation_points: int = 101,
    ) -> BinaryFunction:
        """
        Create balanced t-norm T_B from a t-norm.

        The t-conorm is always generated as the standard dual:
        S(x,y)=1-T(1-x,1-y).
        """

        def t_conorm(x: float, y: float) -> float:
            x = _unit_interval(x, "x")
            y = _unit_interval(y, "y")
            return 1.0 - t_norm(1.0 - x, 1.0 - y)

        return BalancedFuzzyOperators.balanced_t_norm_from_dual_pair(
            t_norm,
            t_conorm,
            validate=validate,
            validation_points=validation_points,
        )

    @staticmethod
    def balanced_t_norm_from_dual_pair(
        t_norm: BinaryFunction,
        t_conorm: BinaryFunction,
        *,
        validate: bool = True,
        validation_points: int = 101,
    ) -> BinaryFunction:
        """
        Create balanced t-norm T_B from a t-norm and its verified dual t-conorm.
        """

        numerically_checked_properties = {
            "empirically_dual_on_grid": False,
            "empirically_maps_balanced_square_to_interval_on_grid": False,
            "empirically_commutative_on_grid": False,
            "empirically_associative_on_grid": False,
            "empirically_non_decreasing_on_grid": False,
            "empirically_boundary_condition_on_grid": False,
            "empirically_symmetric_on_grid": False,
            "empirically_balanced_t_norm_on_grid": False,
        }
        if validate:
            _validate_dual_t_norm_pair(t_norm, t_conorm, validation_points)
            numerically_checked_properties["empirically_dual_on_grid"] = True

        def balanced_t_norm(x: float, y: float) -> float:
            x = _balanced_interval(x, "x")
            y = _balanced_interval(y, "y")

            if x <= 0.0 and y <= 0.0:
                result = t_conorm(x + 1.0, y + 1.0) - 1.0
            elif x >= 0.0 and y >= 0.0:
                result = t_norm(x, y)
            else:
                result = 0.0

            return _clamp_balanced(result, "T_B(x,y)")

        if validate:
            numerically_checked_properties.update(
                _validate_balanced_t_norm_axioms(
                    balanced_t_norm,
                    validation_points,
                )
            )

        balanced_t_norm.numerically_checked_properties = numerically_checked_properties
        balanced_t_norm.verified_properties = numerically_checked_properties
        return attach_guarantee(balanced_t_norm, BY_CONSTRUCTION, validate=validate)

    @staticmethod
    def create_balanced_t_norm(
        t_norm: BinaryFunction,
        t_conorm: BinaryFunction | None = None,
        *,
        validate: bool = True,
        validation_points: int = 101,
    ) -> BinaryFunction:
        """
        Backward-compatible wrapper.

        Without t_conorm it calls balanced_t_norm_from_t_norm. With t_conorm it
        calls balanced_t_norm_from_dual_pair and validates duality.
        """

        if t_conorm is None:
            return BalancedFuzzyOperators.balanced_t_norm_from_t_norm(
                t_norm,
                validate=validate,
                validation_points=validation_points,
            )
        return BalancedFuzzyOperators.balanced_t_norm_from_dual_pair(
            t_norm,
            t_conorm,
            validate=validate,
            validation_points=validation_points,
        )

    @staticmethod
    def transport_nullnorm(
        nullnorm: Nullnorm,
        f: NumberFunction,
        f_inverse: NumberFunction,
        *,
        validate: bool = True,
        validation_points: int = 401,
    ) -> BinaryFunction:
        """Transport a nullnorm from [0,1] to [-1,1] without balanced-symmetry claim."""

        operation = _transport_nullnorm(
            nullnorm,
            f,
            f_inverse,
            validate=validate,
            validation_points=validation_points,
        )
        return attach_guarantee(operation, BY_CONSTRUCTION, validate=validate)

    @staticmethod
    def balanced_t_norm_from_nullnorm(
        nullnorm: Nullnorm,
        f: NumberFunction,
        f_inverse: NumberFunction,
        *,
        validate: bool = True,
        validation_points: int = 401,
    ) -> BinaryFunction:
        """Construct a balanced t-norm from a nullnorm and validate balanced symmetry."""

        operation = _balanced_t_norm_from_nullnorm(
            nullnorm,
            f,
            f_inverse,
            validate=validate,
            validation_points=validation_points,
        )
        if validate:
            operation.numerically_checked_properties = _validate_balanced_t_norm_axioms(
                operation,
                validation_points,
            )
            operation.verified_properties = operation.numerically_checked_properties
        return attach_guarantee(operation, BY_CONSTRUCTION, validate=validate)


def _validate_transport_to_balanced_interval(
    uninorm_generator: AdditiveGeneratorOfRepresentableUninorm,
    f: NumberFunction,
    f_inverse: NumberFunction,
    *,
    balanced_negation: NumberFunction | BalancedFuzzyNegation | None,
    unit_negation: NumberFunction | FuzzyNegation | None,
    validation_points: int,
) -> dict[str, bool]:
    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    e = uninorm_generator.e
    boundary_checks = [
        ("f(-1)", f(-1.0), 0.0),
        ("f(0)", f(0.0), e),
        ("f(1)", f(1.0), 1.0),
        ("f_inverse(0)", f_inverse(0.0), -1.0),
        ("f_inverse(e)", f_inverse(e), 0.0),
        ("f_inverse(1)", f_inverse(1.0), 1.0),
    ]
    for label, actual, expected in boundary_checks:
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
    n_unit: NumberFunction
    if unit_negation is None:
        n_unit = lambda x: 1.0 - x
    else:
        n_unit = unit_negation if callable(unit_negation) else unit_negation.n

    previous_fx = None
    for x in xs:
        fx = _finite(f(x), f"f({x})")
        _unit_interval_with_tolerance(fx, f"f({x})")
        if previous_fx is not None and fx <= previous_fx + EPS:
            raise ValueError("f must be strictly increasing on [-1,1].")
        previous_fx = fx

        roundtrip = _finite(f_inverse(fx), f"f_inverse(f({x}))")
        if abs(roundtrip - x) > 1e-6:
            raise ValueError(
                f"f_inverse must invert f; f_inverse(f({x}))={roundtrip!r}."
            )

    previous_x = None
    for u in us:
        x = _finite(f_inverse(u), f"f_inverse({u})")
        _balanced_interval_with_tolerance(x, f"f_inverse({u})")
        if previous_x is not None and x <= previous_x + EPS:
            raise ValueError("f_inverse must be strictly increasing on [0,1].")
        previous_x = x

        roundtrip = _finite(f(x), f"f(f_inverse({u}))")
        if abs(roundtrip - u) > 1e-6:
            raise ValueError(
                f"f must invert f_inverse; f(f_inverse({u}))={roundtrip!r}."
            )

    if balanced_negation is None and unit_negation is None:
        for x in xs:
            left = _finite(f(-x), f"f({-x})")
            right = 1.0 - _finite(f(x), f"f({x})")
            if abs(left - right) > 1e-6:
                raise ValueError(
                    "f must satisfy the standard symmetry f(-x)=1-f(x), "
                    "or pass balanced_negation and unit_negation explicitly."
                )
    elif balanced_negation is not None and unit_negation is not None:
        n_balanced = balanced_negation if callable(balanced_negation) else balanced_negation.n
        for x in xs:
            left = _finite(f(n_balanced(x)), f"f(N_B({x}))")
            right = _finite(n_unit(f(x)), f"n(f({x}))")
            if abs(left - right) > 1e-6:
                raise ValueError(
                    f"Transport symmetry failed: f(N_B({x}))={left!r}, "
                    f"n(f({x}))={right!r}."
                )
    else:
        raise ValueError(
            "balanced_negation and unit_negation must be passed together "
            "when custom symmetry is validated."
        )

    for x in us:
        nx = _unit_interval_with_tolerance(n_unit(x), f"n({x})")
        ux = uninorm_generator.u(x)
        unx = uninorm_generator.u(nx)
        if not _are_opposites_in_extended_reals(unx, ux):
            raise ValueError(
                "Uninorm generator antisymmetry failed: "
                f"u(n({x}))={unx!r}, -u({x})={-ux!r}."
            )

    transported_neutral = f_inverse(e)
    if abs(transported_neutral) > EPS:
        raise ValueError(
            f"Transported neutral element must be 0, got {transported_neutral!r}."
        )

    for x in xs:
        fx = f(x)
        left_neutral = f_inverse(uninorm_generator.uninorm(fx, e))
        right_neutral = f_inverse(uninorm_generator.uninorm(e, fx))
        if abs(left_neutral - x) > 1e-6 or abs(right_neutral - x) > 1e-6:
            raise ValueError(
                "Transported operation must have neutral element 0; "
                f"got S_B({x},0)={left_neutral!r}, S_B(0,{x})={right_neutral!r}."
            )
    return {
        "empirically_transport_valid_on_grid": True,
        "empirically_uninorm_generator_antisymmetric_on_grid": True,
        "empirically_symmetric_on_grid": True,
    }


def _validate_dual_t_norm_pair(
    t_norm: BinaryFunction,
    t_conorm: BinaryFunction,
    validation_points: int,
) -> None:
    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    points = [
        index / (validation_points - 1)
        for index in range(validation_points)
    ]
    for x in points:
        for y in points:
            tx = _unit_interval_with_tolerance(t_norm(x, y), f"T({x}, {y})")
            sx = _unit_interval_with_tolerance(t_conorm(x, y), f"S({x}, {y})")
            expected = 1.0 - _unit_interval_with_tolerance(
                t_norm(1.0 - x, 1.0 - y),
                f"T({1.0 - x}, {1.0 - y})",
            )
            if abs(sx - expected) > 1e-6:
                raise ValueError(
                    "t_conorm must be dual to t_norm: "
                    f"S({x}, {y})={sx!r}, expected {expected!r}."
                )
            if x == 1.0 and abs(tx - y) > 1e-6:
                raise ValueError(f"t_norm boundary failed: T(1,{y})={tx!r}.")
            if y == 1.0 and abs(tx - x) > 1e-6:
                raise ValueError(f"t_norm boundary failed: T({x},1)={tx!r}.")
            if x == 0.0 and abs(sx - y) > 1e-6:
                raise ValueError(f"t_conorm boundary failed: S(0,{y})={sx!r}.")
            if y == 0.0 and abs(sx - x) > 1e-6:
                raise ValueError(f"t_conorm boundary failed: S({x},0)={sx!r}.")


def _validate_balanced_t_norm_symmetry(
    balanced_t_norm: BinaryFunction,
    validation_points: int,
) -> None:
    points = [
        -1.0 + 2.0 * index / (validation_points - 1)
        for index in range(validation_points)
    ]
    for x in points:
        for y in points:
            value = _balanced_interval_with_tolerance(
                balanced_t_norm(x, y),
                f"P({x}, {y})",
            )
            opposite = _balanced_interval_with_tolerance(
                balanced_t_norm(-x, -y),
                f"P({-x}, {-y})",
            )
            if abs(value + opposite) > 1e-6:
                raise ValueError(
                    "balanced t-norm symmetry failed: "
                    f"P({x}, {y})={value!r}, -P({-x}, {-y})={-opposite!r}."
                )


def _validate_balanced_t_norm_axioms(
    balanced_t_norm: BinaryFunction,
    validation_points: int,
) -> dict[str, bool]:
    """
    Validate balanced t-norm axioms on a finite grid.

    The boundary condition used by the implemented balanced construction is:
        P(x, 1)=P(1, x)=max(x, 0)
        P(x,-1)=P(-1,x)=min(x, 0)
    """

    if validation_points < 3:
        raise ValueError("validation_points must be at least 3.")

    points = [
        -1.0 + 2.0 * index / (validation_points - 1)
        for index in range(validation_points)
    ]
    axiom_points = _validation_subset(points, limit=21)
    associativity_points = _validation_subset(points, limit=11)

    values: dict[tuple[float, float], float] = {}
    maps_balanced_square = True
    for x in axiom_points:
        for y in axiom_points:
            try:
                values[(x, y)] = _balanced_interval_with_tolerance(
                    balanced_t_norm(x, y),
                    f"P({x}, {y})",
                )
            except (ArithmeticError, ValueError, OverflowError):
                maps_balanced_square = False
                values[(x, y)] = float("nan")

    commutative = maps_balanced_square
    for x in axiom_points:
        for y in axiom_points:
            if abs(values[(x, y)] - values[(y, x)]) > 1e-6:
                commutative = False
                break

    non_decreasing = maps_balanced_square
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

    associative = maps_balanced_square
    for x in associativity_points:
        for y in associativity_points:
            for z in associativity_points:
                try:
                    left = _balanced_interval_with_tolerance(
                        balanced_t_norm(balanced_t_norm(x, y), z),
                        f"P(P({x}, {y}), {z})",
                    )
                    right = _balanced_interval_with_tolerance(
                        balanced_t_norm(x, balanced_t_norm(y, z)),
                        f"P({x}, P({y}, {z}))",
                    )
                except (ArithmeticError, ValueError, OverflowError):
                    associative = False
                    break
                if abs(left - right) > 1e-6:
                    associative = False
                    break

    boundary_condition = maps_balanced_square
    for x in axiom_points:
        expected_top = max(x, 0.0)
        expected_bottom = min(x, 0.0)
        checks = (
            (balanced_t_norm(x, 1.0), expected_top, f"P({x}, 1)"),
            (balanced_t_norm(1.0, x), expected_top, f"P(1, {x})"),
            (balanced_t_norm(x, -1.0), expected_bottom, f"P({x}, -1)"),
            (balanced_t_norm(-1.0, x), expected_bottom, f"P(-1, {x})"),
        )
        for actual, expected, label in checks:
            actual = _balanced_interval_with_tolerance(actual, label)
            if abs(actual - expected) > 1e-6:
                boundary_condition = False
                break

    symmetric = maps_balanced_square
    for x in axiom_points:
        for y in axiom_points:
            value = values[(x, y)]
            opposite = values.get((-x, -y))
            if opposite is None:
                opposite = _balanced_interval_with_tolerance(
                    balanced_t_norm(-x, -y),
                    f"P({-x}, {-y})",
                )
            if abs(value + opposite) > 1e-6:
                symmetric = False
                break

    valid = (
        maps_balanced_square
        and commutative
        and associative
        and non_decreasing
        and boundary_condition
        and symmetric
    )
    properties = {
        "empirically_maps_balanced_square_to_interval_on_grid": maps_balanced_square,
        "empirically_commutative_on_grid": commutative,
        "empirically_associative_on_grid": associative,
        "empirically_non_decreasing_on_grid": non_decreasing,
        "empirically_boundary_condition_on_grid": boundary_condition,
        "empirically_symmetric_on_grid": symmetric,
        "empirically_balanced_t_norm_on_grid": valid,
    }
    if not valid:
        failed = [name for name, passed in properties.items() if not passed]
        raise ValueError(
            "balanced t-norm validation failed on the grid; failed properties: "
            f"{', '.join(failed)}."
        )
    return properties


def _are_opposites_in_extended_reals(left: float, right: float) -> bool:
    if isinf(left) or isinf(right):
        return isinf(left) and isinf(right) and left == -right
    _finite(left, "left")
    _finite(right, "right")
    return abs(left + right) <= 1e-6


def _finite(value: float, name: str) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _unit_interval_with_tolerance(x: float, name: str) -> float:
    _finite(x, name)
    if x < -EPS or x > 1.0 + EPS:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _balanced_interval_with_tolerance(x: float, name: str) -> float:
    _finite(x, name)
    if x < -1.0 - EPS or x > 1.0 + EPS:
        raise ValueError(f"{name} must be in [-1, 1], got {x!r}.")
    if x <= -1.0:
        return -1.0
    if x >= 1.0:
        return 1.0
    if abs(x) <= EPS:
        return 0.0
    return x


def _validation_subset(points: list[float], *, limit: int) -> list[float]:
    if len(points) <= limit:
        return points
    indexes = {
        round(index * (len(points) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [points[index] for index in sorted(indexes)]


balanced_fuzzy_operators = BalancedFuzzyOperators
