"""
An additive generator of representable uninorm, based on section 4 of
2025_IWIFSGN_Warsaw.pdf.

For the standard negation n(x)=1-x and a strict t-norm additive generator t,
this module constructs an additive generator u of a representable uninorm with
neutral element e in (0, 1):

    u(x) = -t(x/e)                         for x in [0, e]
    u(x) =  t(1 - (x-e)/(1-e))             for x in (e, 1]

The second branch is s((x-e)/(1-e)) for the dual t-conorm generator
s(x)=t(n(x)).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isinf

from .operator_guarantees import BY_CONSTRUCTION
from .tnorm_additive_generators import AdditiveGenerator


def _unit_interval(x: float, name: str) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x!r}")
    return x


def _neutral(e: float) -> float:
    if not 0.0 < e < 1.0:
        raise ValueError(f"e must be in (0, 1), got {e!r}")
    return e


@dataclass(frozen=True)
class AdditiveGeneratorOfRepresentableUninorm:
    """Additive generator u of a representable uninorm and its inverse."""

    t_generator: AdditiveGenerator
    e: float
    guarantee: str = BY_CONSTRUCTION

    def __post_init__(self) -> None:
        _neutral(self.e)
        if not isinf(self.t_generator.f0_plus):
            raise ValueError(
                "An additive generator of a representable uninorm requires a strict "
                "t-norm generator, so t(0+) must be +inf."
            )

    @property
    def name(self) -> str:
        return (
            "AdditiveGeneratorOfRepresentableUninorm"
            f"({self.t_generator.name}, e={self.e:g})"
        )

    def u(self, x: float) -> float:
        """Evaluate the uninorm additive generator u(x)."""

        x = _unit_interval(x, "x")
        if x <= self.e:
            return -self.t_generator(x / self.e)
        return self.t_generator(1.0 - (x - self.e) / (1.0 - self.e))

    def inverse(self, z: float) -> float:
        """Evaluate u^{-1}(z)."""

        if z <= 0.0:
            return self.e * self.t_generator.inverse(-z)
        return self.e + (1.0 - self.e) * (1.0 - self.t_generator.inverse(z))

    def uninorm(self, x: float, y: float, *, infinity_convention: str = "disjunctive") -> float:
        """
        Representable uninorm U(x,y)=u^{-1}(u(x)+u(y)).

        The sum (+inf)+(-inf) appears only at opposite corners (0,1) and (1,0).
        Use infinity_convention="disjunctive" to map it to +inf, or
        "conjunctive" to map it to -inf.
        """

        x = _unit_interval(x, "x")
        y = _unit_interval(y, "y")
        ux = self.u(x)
        uy = self.u(y)

        if {ux, uy} == {-inf, inf}:
            if infinity_convention == "disjunctive":
                return 1.0
            if infinity_convention == "conjunctive":
                return 0.0
            raise ValueError(
                "infinity_convention must be 'disjunctive' or 'conjunctive'."
            )

        return self.inverse(ux + uy)


def additive_generator_of_representable_uninorm(
    t_generator: AdditiveGenerator,
    e: float,
) -> AdditiveGeneratorOfRepresentableUninorm:
    """
    Build an additive generator u of a representable uninorm.

    Parameters:
        t_generator: strict t-norm additive generator t, e.g. product(),
            aczel_alsina(p), dombi(p), hamacher(p), or Schweizer-Sklar p<0.
        e: neutral element of the resulting uninorm, e in (0, 1).
    """

    return AdditiveGeneratorOfRepresentableUninorm(t_generator=t_generator, e=e)
