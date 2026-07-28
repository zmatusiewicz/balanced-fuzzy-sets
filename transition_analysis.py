"""Stage-based transition analysis using rules from prefix-notation files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Mapping

from rule_builder import OperatorFunction, RuleBuilder


RuleFunction = Callable[..., float]
RULE_VARIABLE_ORDER = tuple("xyzabcdefghijklmnopqrstuvw")


@dataclass(frozen=True)
class RuleCall:
    """One call such as x1:=Rule1(x1,x3)."""

    output_place: str
    rule_name: str
    arguments: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class AnalysisStage:
    """One stage such as Stage1. x1=0.5, x2=0.7; Rule1(x1,x3)."""

    name: str
    values: dict[str, float]
    calls: tuple[RuleCall, ...]


@dataclass(frozen=True)
class RuleResult:
    """Evaluated rule value inside one stage."""

    stage: str
    rule_name: str
    arguments: tuple[str, ...]
    output_place: str
    priority: int
    value: float


def parse_stage(text: str) -> AnalysisStage:
    """
    Parse one stage.

    Example:
        Stage1. x1=0.5, x2=0.7, x3=-0.2; Rule1(x1,x3)
    """

    header, body = text.split(".", 1)
    name = header.strip()
    parts = [part.strip() for part in body.split(";") if part.strip()]
    if not parts:
        raise ValueError(f"Stage {name!r} has no assignments.")

    values = _parse_assignments(parts[0])
    calls = tuple(
        call
        for _, call in sorted(
            (
                (index, _parse_rule_call(part, default_priority=index))
                for index, part in enumerate(parts[1:], start=1)
            ),
            key=lambda item: (item[1].priority, item[0]),
        )
    )
    return AnalysisStage(name=name, values=values, calls=calls)


def parse_stages(text: str) -> list[AnalysisStage]:
    """Parse many stages separated by new lines."""

    stages = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            stages.append(parse_stage(line))
    return stages


def load_rule(rule_name: str, rules_dir: str | Path = "rules") -> str:
    """Load a compact prefix-notation rule from rules_dir/rule_name."""

    path = Path(rules_dir) / rule_name
    if not path.exists():
        raise FileNotFoundError(f"Rule file does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_rules(rules_dir: str | Path = "rules") -> dict[str, str]:
    """Load all rule files from a directory."""

    directory = Path(rules_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Rules directory does not exist: {directory}")

    return {
        path.name: path.read_text(encoding="utf-8").strip()
        for path in directory.iterdir()
        if path.is_file()
    }


def evaluate_stage(
    stage: AnalysisStage,
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
    rule_builder: RuleBuilder | None = None,
) -> list[RuleResult]:
    """Evaluate all rule calls in one stage."""

    builder = RuleBuilder() if rule_builder is None else rule_builder
    results = []
    values = dict(stage.values)

    for call in stage.calls:
        rule_code = load_rule(call.rule_name, rules_dir)
        rule_node = builder.parse(rule_code)
        rule_variables = _ordered_rule_variables(rule_node.variables())

        if len(call.arguments) != len(rule_variables):
            raise ValueError(
                f"{call.rule_name} needs exactly {len(rule_variables)} arguments "
                f"for variables {rule_variables}, got {len(call.arguments)}: "
                f"{call.arguments}."
            )

        mapped_values = {}
        for variable, argument_name in zip(rule_variables, call.arguments):
            if argument_name not in values:
                raise KeyError(
                    f"Missing value {argument_name!r} in stage {stage.name!r}."
                )
            mapped_values[variable] = values[argument_name]

        value = rule_node.evaluate(mapped_values, operators)
        output_place = call.output_place
        values[output_place] = value
        results.append(
            RuleResult(
                stage=stage.name,
                rule_name=call.rule_name,
                arguments=call.arguments,
                output_place=output_place,
                priority=call.priority,
                value=value,
            )
        )

    return results


def evaluate_analysis(
    text: str,
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
) -> list[RuleResult]:
    """Parse and evaluate all stages from text."""

    results = []
    for stage in parse_stages(text):
        results.extend(evaluate_stage(stage, operators, rules_dir=rules_dir))
    return results


def format_results(results: list[RuleResult]) -> str:
    """Format analysis results as readable text."""

    lines = []
    for result in results:
        args = ", ".join(result.arguments)
        lines.append(
            f"{result.stage}: p={result.priority} {result.output_place}:="
            f"{result.rule_name}({args}) "
            f"= {result.value:.12f}"
        )
    return "\n".join(lines)


def _parse_assignments(text: str) -> dict[str, float]:
    values = {}
    for assignment in text.split(","):
        assignment = assignment.strip()
        if not assignment:
            continue
        variable, raw_value = assignment.split("=", 1)
        values[variable.strip()] = float(raw_value.strip())
    return values


def _parse_rule_call(text: str, *, default_priority: int) -> RuleCall:
    priority = default_priority
    priority_match = re.fullmatch(r"(\d+)\s*:\s*(.+)", text.strip())
    if priority_match is not None:
        priority = int(priority_match.group(1))
        text = priority_match.group(2).strip()

    output_place = None
    if ":=" in text:
        output_place, text = [part.strip() for part in text.split(":=", 1)]
    if not output_place:
        raise ValueError(
            f"Rule call must use explicit output assignment, e.g. x1:=Rule1(x1): {text!r}"
        )

    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\(([^)]*)\)", text)
    if match is None:
        raise ValueError(f"Invalid rule call: {text!r}")

    arguments = tuple(
        argument.strip()
        for argument in match.group(2).split(",")
        if argument.strip()
    )
    return RuleCall(
        output_place=output_place,
        rule_name=match.group(1),
        arguments=arguments,
        priority=priority,
    )


def _ordered_rule_variables(variables: set[str]) -> tuple[str, ...]:
    known = [variable for variable in RULE_VARIABLE_ORDER if variable in variables]
    other = sorted(variables - set(known))
    return tuple(known + other)
