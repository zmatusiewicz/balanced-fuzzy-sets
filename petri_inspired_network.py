"""Petri-inspired transition network representation for staged takeover-rule analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from transition_analysis import RULE_VARIABLE_ORDER, RuleCall, load_rule, parse_stage
from rule_builder import OperatorFunction, RuleBuilder


@dataclass(frozen=True)
class Place:
    """Petri-inspired place represented as a circle."""

    name: str
    value: float
    is_leaf: bool


@dataclass(frozen=True)
class Transition:
    """Petri-inspired transition represented as a rule application."""

    name: str
    rule_name: str
    priority: int
    rule_code: str
    rule_infix: str
    input_places: tuple[str, ...]
    output_place: str
    output_value: float


@dataclass(frozen=True)
class Arc:
    """Directed Petri-inspired arc."""

    source: str
    target: str
    label: str = ""


@dataclass(frozen=True)
class PetriInspiredNetwork:
    """Petri-inspired transition network built from one analysis stage."""

    stage_name: str
    places: dict[str, Place]
    transitions: list[Transition]
    arcs: list[Arc]


def build_petri_inspired_network_from_stage_text(
    text: str,
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
) -> PetriInspiredNetwork:
    """Build a Petri-inspired transition network from text such as Step2."""

    stage = parse_stage(text)
    return build_petri_inspired_network(
        stage.name,
        stage.values,
        list(stage.calls),
        operators,
        rules_dir=rules_dir,
    )


def build_petri_inspired_network_from_step_file(
    path: str | Path,
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
) -> PetriInspiredNetwork:
    """Build a Petri-inspired transition network from a step file."""

    return build_petri_inspired_network_from_stage_text(
        Path(path).read_text(encoding="utf-8"),
        operators,
        rules_dir=rules_dir,
    )


def build_petri_inspired_network_from_process_steps(
    paths: list[str | Path],
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
    stage_name: str = "Multi-step Petri-inspired transition network",
) -> PetriInspiredNetwork:
    """
    Build one layered Petri-inspired transition network for consecutive process steps.

    The first file initializes the process state. Values in later files are not
    treated as new independent initials; they must match the current state.
    """

    stages = []
    for path in paths:
        stage = parse_stage(Path(path).read_text(encoding="utf-8"))
        stages.append(stage)

    if not stages:
        raise ValueError("At least one step file is needed.")

    builder = RuleBuilder()
    values = dict(stages[0].values)
    variables = sorted(values, key=_place_sort_key)
    places = {
        _layered_place_name(name, 1): Place(
            name=_layered_place_name(name, 1),
            value=value,
            is_leaf=True,
        )
        for name, value in values.items()
    }
    transitions = []
    arcs = []
    layer = 1

    for stage in stages:
        if stage is not stages[0]:
            _merge_stage_values_into_process_state(stage.name, stage.values, values, variables)
            for name in variables:
                place_name = _layered_place_name(name, layer)
                if place_name not in places:
                    places[place_name] = Place(
                        name=place_name,
                        value=values[name],
                        is_leaf=True,
                    )

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
                    raise KeyError(f"Missing place {argument_name!r} before {call.rule_name}.")
                mapped_values[variable] = values[argument_name]

            output_variable = call.output_place
            output_value = rule_node.evaluate(mapped_values, operators)
            next_values = dict(values)
            next_values[output_variable] = output_value
            for name in next_values:
                if name not in variables:
                    variables.append(name)
                    variables.sort(key=_place_sort_key)

            next_layer = layer + 1
            for name in variables:
                place_name = _layered_place_name(name, next_layer)
                places[place_name] = Place(
                    name=place_name,
                    value=next_values[name],
                    is_leaf=False,
                )

            transition_name = f"t{len(transitions) + 1}"
            input_places = tuple(_layered_place_name(argument, layer) for argument in call.arguments)
            output_place = _layered_place_name(output_variable, next_layer)
            transitions.append(
                Transition(
                    name=transition_name,
                    rule_name=call.rule_name,
                    priority=call.priority,
                    rule_code=rule_code,
                    rule_infix=rule_node.to_infix(),
                    input_places=input_places,
                    output_place=output_place,
                    output_value=output_value,
                )
            )

            for place_name in input_places:
                arcs.append(Arc(source=place_name, target=transition_name))
            for name in variables:
                arcs.append(Arc(source=transition_name, target=_layered_place_name(name, next_layer)))

            values = next_values
            layer = next_layer

    return PetriInspiredNetwork(
        stage_name=stage_name,
        places=places,
        transitions=transitions,
        arcs=arcs,
    )


def build_petri_inspired_networks_from_independent_scenarios(
    paths: list[str | Path],
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
) -> list[PetriInspiredNetwork]:
    """Build one independent Petri-inspired transition network per step/scenario file."""

    return [
        build_petri_inspired_network_from_step_file(path, operators, rules_dir=rules_dir)
        for path in paths
    ]


def build_petri_inspired_network(
    stage_name: str,
    initial_values: Mapping[str, float],
    rule_calls: list[str | RuleCall],
    operators: Mapping[str, OperatorFunction],
    *,
    rules_dir: str | Path = "rules",
) -> PetriInspiredNetwork:
    """
    Build a Petri-inspired transition network from places and rule calls.

    Rule calls must be written with explicit output assignment,
    e.g. x5:=Rule1(x1).
    """

    builder = RuleBuilder()
    values = dict(initial_values)
    places = {
        name: Place(name=name, value=value, is_leaf=True)
        for name, value in sorted(values.items(), key=lambda item: _place_sort_key(item[0]))
    }
    transitions = []
    arcs = []

    for index, raw_call in enumerate(rule_calls, start=1):
        output_name, rule_name, arguments, priority = _normalize_rule_application(raw_call)
        if output_name is None:
            raise ValueError(
                f"{rule_name} call must use explicit output assignment, "
                f"e.g. x1:=Rule1(x1)."
            )

        rule_code = load_rule(rule_name, rules_dir)
        rule_node = builder.parse(rule_code)
        rule_variables = _ordered_rule_variables(rule_node.variables())
        if len(arguments) != len(rule_variables):
            raise ValueError(
                f"{rule_name} needs exactly {len(rule_variables)} arguments "
                f"for variables {rule_variables}, got {len(arguments)}: {arguments}."
            )

        mapped_values = {}
        for variable, argument_name in zip(rule_variables, arguments):
            if argument_name not in values:
                raise KeyError(f"Missing place {argument_name!r} before {rule_name}.")
            mapped_values[variable] = values[argument_name]

        output_value = rule_node.evaluate(mapped_values, operators)
        values[output_name] = output_value
        places[output_name] = Place(
            name=output_name,
            value=output_value,
            is_leaf=False,
        )

        transition_name = f"t{index}"
        transition = Transition(
            name=transition_name,
            rule_name=rule_name,
            priority=priority,
            rule_code=rule_code,
            rule_infix=rule_node.to_infix(),
            input_places=arguments,
            output_place=output_name,
            output_value=output_value,
        )
        transitions.append(transition)

        for argument in arguments:
            arcs.append(Arc(source=argument, target=transition_name))
        arcs.append(
            Arc(
                source=transition_name,
                target=output_name,
                label=f"{output_name}:={rule_name}({', '.join(arguments)})",
            )
        )

    return PetriInspiredNetwork(
        stage_name=stage_name,
        places=places,
        transitions=transitions,
        arcs=arcs,
    )


def draw_petri_inspired_network(
    network: PetriInspiredNetwork,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Draw a Petri-inspired transition network with places as circles and transitions as rectangles."""

    positions = _layout_positions(network)
    width = max(11, max((x for x, _ in positions.values()), default=0.0) + 1.2)
    height = max(4.5, max((y for _, y in positions.values()), default=0.0) + 1.0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    for arc in network.arcs:
        start = positions[arc.source]
        end = positions[arc.target]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#374151"},
            zorder=1,
        )

    for place in network.places.values():
        x, y = positions[place.name]
        circle = plt.Circle(
            (x, y),
            0.16,
            facecolor="#f3f7ff" if place.is_leaf else "#eef7ef",
            edgecolor="#1f2937",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y + 0.015,
            _display_place_name(place.name),
            ha="center",
            va="center",
            fontsize=9,
            zorder=4,
        )
        ax.text(
            x,
            y - 0.105,
            f"{place.value:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )

    for transition in network.transitions:
        x, y = positions[transition.name]
        box = plt.Rectangle(
            (x - 0.18, y - 0.11),
            0.36,
            0.22,
            facecolor="#fff7ed",
            edgecolor="#9a3412",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, transition.rule_name, ha="center", va="center", fontsize=9, zorder=4)
    ax.set_xlim(-0.5, max((x for x, _ in positions.values()), default=0.0) + 0.7)
    ax.set_ylim(-0.4, max(1.0, max((y for _, y in positions.values()), default=1.0)) + 0.5)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def petri_inspired_network_summary(network: PetriInspiredNetwork) -> str:
    """Return readable text summary of places and transitions."""

    lines = [network.stage_name, "Places:"]
    for place in sorted(network.places.values(), key=lambda item: _place_sort_key(item.name)):
        kind = "leaf" if place.is_leaf else "computed"
        lines.append(f"- {_display_place_name(place.name)} = {place.value:.12f} ({kind})")

    lines.append("Transitions:")
    for transition in network.transitions:
        args = ", ".join(_display_place_name(place) for place in transition.input_places)
        lines.append(
            f"- p={transition.priority} {_display_place_name(transition.output_place)}:="
            f"{transition.rule_name}({args}) "
            f"= {transition.output_value:.12f}"
        )
    return "\n".join(lines)


def _normalize_rule_application(
    call: str | RuleCall,
) -> tuple[str | None, str, tuple[str, ...], int]:
    if isinstance(call, RuleCall):
        return call.output_place, call.rule_name, call.arguments, call.priority
    priority = 1
    text = call.strip()
    priority_match = re.fullmatch(r"(\d+)\s*:\s*(.+)", text)
    if priority_match is not None:
        priority = int(priority_match.group(1))
        text = priority_match.group(2).strip()
    output_name, rule_name, arguments = _parse_rule_application(text)
    return output_name, rule_name, arguments, priority


def _parse_rule_application(text: str) -> tuple[str | None, str, tuple[str, ...]]:
    text = text.strip()
    output_name = None
    if ":=" in text:
        output_name, text = [part.strip() for part in text.split(":=", 1)]

    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\(([^)]*)\)", text)
    if match is None:
        raise ValueError(f"Invalid rule call: {text!r}")

    arguments = tuple(
        argument.strip()
        for argument in match.group(2).split(",")
        if argument.strip()
    )
    return output_name, match.group(1), arguments


def _layout_positions(network: PetriInspiredNetwork) -> dict[str, tuple[float, float]]:
    layered_positions = _layout_layered_positions(network)
    if layered_positions is not None:
        return layered_positions

    positions: dict[str, tuple[float, float]] = {}
    leaf_places = [place for place in network.places.values() if place.is_leaf]
    computed_places = [place for place in network.places.values() if not place.is_leaf]

    for index, place in enumerate(sorted(leaf_places, key=lambda item: _place_sort_key(item.name))):
        positions[place.name] = (0.0, len(leaf_places) - index)

    for index, transition in enumerate(network.transitions):
        y = len(network.transitions) - index
        positions[transition.name] = (1.45, y)

    for index, place in enumerate(computed_places):
        y = len(network.transitions) - index
        positions[place.name] = (2.9, y)

    return positions


def _merge_stage_values_into_process_state(
    stage_name: str,
    stage_values: Mapping[str, float],
    current_values: dict[str, float],
    variables: list[str],
) -> None:
    for name, value in stage_values.items():
        if name in current_values:
            if abs(current_values[name] - value) > 1e-9:
                raise ValueError(
                    f"{stage_name} declares {name}={value!r}, but current process "
                    f"state has {name}={current_values[name]!r}. Use "
                    "build_petri_inspired_networks_from_independent_scenarios for independent "
                    "scenarios, or make consecutive step values match the state."
                )
        else:
            current_values[name] = value
            if name not in variables:
                variables.append(name)
                variables.sort(key=_place_sort_key)


def _layout_layered_positions(network: PetriInspiredNetwork) -> dict[str, tuple[float, float]] | None:
    parsed_places = {
        place.name: _parse_layered_place_name(place.name)
        for place in network.places.values()
    }
    if any(parsed is None for parsed in parsed_places.values()):
        return None

    variables = sorted({parsed[0] for parsed in parsed_places.values() if parsed}, key=_place_sort_key)
    variable_y = {
        variable: float(len(variables) - index)
        for index, variable in enumerate(variables)
    }
    spacing = 1.9
    positions: dict[str, tuple[float, float]] = {}
    for name, parsed in parsed_places.items():
        if parsed is None:
            continue
        variable, layer = parsed
        positions[name] = ((layer - 1) * spacing, variable_y[variable])

    for transition in network.transitions:
        parsed_output = _parse_layered_place_name(transition.output_place)
        if parsed_output is None:
            return None
        output_variable, output_layer = parsed_output
        positions[transition.name] = (
            (output_layer - 1) * spacing - spacing / 2.0,
            variable_y[output_variable],
        )

    return positions


def _place_sort_key(name: str) -> tuple[int, str]:
    layered = _parse_layered_place_name(name)
    if layered is not None:
        variable, layer = layered
        variable_key = _place_sort_key(variable)
        return (layer, f"{variable_key[0]:09d}_{variable_key[1]}")

    match = re.fullmatch(r"x(\d+)", name)
    if match is None:
        return (10**9, name)
    return (int(match.group(1)), name)


def _display_place_name(name: str) -> str:
    if _parse_layered_place_name(name) is not None:
        return name
    match = re.fullmatch(r"Stage(\d+)_(x\d+)", name)
    if match is not None:
        return f"{match.group(2)}({match.group(1)})"
    return name


def _layered_place_name(variable: str, layer: int) -> str:
    return f"{variable}({layer})"


def _parse_layered_place_name(name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(x\d+)\((\d+)\)", name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _ordered_rule_variables(variables: set[str]) -> tuple[str, ...]:
    known = [variable for variable in RULE_VARIABLE_ORDER if variable in variables]
    other = sorted(variables - set(known))
    return tuple(known + other)
