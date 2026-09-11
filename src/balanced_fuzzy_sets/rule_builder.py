"""Build and evaluate fuzzy rules written in prefix notation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

UnaryFunction = Callable[[float], float]
BinaryFunction = Callable[[float, float], float]
OperatorFunction = UnaryFunction | BinaryFunction


DEFAULT_ARITY = {
    "N": 1,
    "R": 1,
    "T": 2,
    "S": 2,
}


@dataclass(frozen=True)
class RuleNode:
    """Expression tree node for a fuzzy rule."""

    symbol: str
    children: tuple[RuleNode, ...] = ()

    @property
    def is_variable(self) -> bool:
        return len(self.children) == 0

    def to_infix(self) -> str:
        """Return readable function notation, e.g. T(S(N(x), y), N(x))."""

        if self.is_variable:
            return self.symbol
        args = ", ".join(child.to_infix() for child in self.children)
        return f"{self.symbol}({args})"

    def variables(self) -> set[str]:
        """Return variables used in the rule."""

        if self.is_variable:
            return {self.symbol}
        result: set[str] = set()
        for child in self.children:
            result.update(child.variables())
        return result

    def evaluate(
        self,
        variables: Mapping[str, float],
        operators: Mapping[str, OperatorFunction],
    ) -> float:
        """Evaluate the rule for given variable values and operator functions."""

        if self.is_variable:
            return variables[self.symbol]

        if self.symbol not in operators:
            raise KeyError(f"Missing operator implementation for {self.symbol!r}.")

        values = [child.evaluate(variables, operators) for child in self.children]
        function = operators[self.symbol]
        return function(*values)


class RuleBuilder:
    """Parser and evaluator for compact Polish notation fuzzy rules."""

    def __init__(self, arity: Mapping[str, int] | None = None) -> None:
        self.arity = dict(DEFAULT_ARITY if arity is None else arity)

    def parse(self, rule: str) -> RuleNode:
        """Parse compact Polish notation into a RuleNode tree."""

        tokens = [token for token in rule if not token.isspace()]
        if not tokens:
            raise ValueError("rule must not be empty.")

        node, position = self._parse_from(tokens, 0)
        if position != len(tokens):
            tail = "".join(tokens[position:])
            raise ValueError(f"Unexpected trailing tokens: {tail!r}.")
        return node

    def to_infix(self, rule: str) -> str:
        """Convert compact Polish notation to readable function notation."""

        return self.parse(rule).to_infix()

    def build(
        self,
        rule: str,
        operators: Mapping[str, OperatorFunction],
    ) -> Callable[..., float]:
        """
        Build a callable from rule notation.

        The returned callable accepts keyword arguments named like variables in the
        rule, e.g. build("TSNxyRX", ops)(x=0.2, y=0.8).
        """

        node = self.parse(rule)

        def evaluator(**variables: float) -> float:
            missing = node.variables() - variables.keys()
            if missing:
                raise KeyError(f"Missing variable values: {sorted(missing)}.")
            extra = variables.keys() - node.variables()
            if extra:
                raise KeyError(f"Unexpected variable values: {sorted(extra)}.")
            return node.evaluate(variables, operators)

        evaluator.__name__ = f"rule_{rule}"
        return evaluator

    def evaluate(
        self,
        rule: str,
        variables: Mapping[str, float],
        operators: Mapping[str, OperatorFunction],
    ) -> float:
        """Parse and evaluate rule in one call."""

        return self.parse(rule).evaluate(variables, operators)

    def _parse_from(self, tokens: list[str], position: int) -> tuple[RuleNode, int]:
        if position >= len(tokens):
            raise ValueError("Unexpected end of rule.")

        symbol = tokens[position]
        position += 1

        if symbol not in self.arity:
            if symbol.islower():
                return RuleNode(symbol), position
            if symbol.isupper():
                raise ValueError(f"Unknown operator {symbol!r}.")
            raise ValueError(f"Unknown token {symbol!r}.")

        children = []
        for _ in range(self.arity[symbol]):
            child, position = self._parse_from(tokens, position)
            children.append(child)
        return RuleNode(symbol, tuple(children)), position


def parse_rule(rule: str, arity: Mapping[str, int] | None = None) -> RuleNode:
    """Parse compact Polish notation into a RuleNode tree."""

    return RuleBuilder(arity).parse(rule)


def rule_to_infix(rule: str, arity: Mapping[str, int] | None = None) -> str:
    """Convert compact Polish notation to readable function notation."""

    return RuleBuilder(arity).to_infix(rule)


def build_rule(
    rule: str,
    operators: Mapping[str, OperatorFunction],
    arity: Mapping[str, int] | None = None,
) -> Callable[..., float]:
    """Build a callable fuzzy rule from compact Polish notation."""

    return RuleBuilder(arity).build(rule, operators)
