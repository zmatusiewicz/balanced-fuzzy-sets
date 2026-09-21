"""Plotting helpers for fuzzy operators and evaluated rule trees."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .rule_builder import OperatorFunction, RuleEvaluationTrace, RuleNode

NumberFunction = Callable[[float], float]
BinaryFunction = Callable[[float, float], float]


def _function_name(function: object, fallback: str) -> str:
    return getattr(function, "name", getattr(function, "__name__", fallback))


def _finalize_figure(
    fig: Figure,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _safe_unary_values(function: NumberFunction, xs: np.ndarray) -> np.ndarray:
    values = []
    for x in xs:
        try:
            values.append(function(float(x)))
        except (ArithmeticError, ValueError, ZeroDivisionError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _break_at_discontinuities(
    xs: np.ndarray,
    ys: np.ndarray,
    discontinuities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if not discontinuities:
        return xs, ys

    new_xs = []
    new_ys = []
    breaks = sorted(point for point in discontinuities if xs[0] < point < xs[-1])

    for index, (x, y) in enumerate(zip(xs, ys)):
        if index > 0:
            previous_x = xs[index - 1]
            for point in breaks:
                if previous_x < point <= x:
                    new_xs.append(point)
                    new_ys.append(np.nan)
                    break
        new_xs.append(x)
        new_ys.append(y)

    return np.asarray(new_xs, dtype=float), np.asarray(new_ys, dtype=float)


def _safe_binary_values(
    function: BinaryFunction,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    values = np.empty_like(xs, dtype=float)
    for index, _ in np.ndenumerate(xs):
        try:
            values[index] = function(float(xs[index]), float(ys[index]))
        except (ArithmeticError, ValueError, ZeroDivisionError):
            values[index] = np.nan
    return values


def _break_binary_at_discontinuities(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    discontinuities_x: Sequence[float],
    discontinuities_y: Sequence[float],
) -> np.ndarray:
    broken = zs.copy()

    for point in discontinuities_x:
        if xs.min() < point < xs.max():
            column = int(np.argmin(np.abs(xs[0, :] - point)))
            broken[:, column] = np.nan

    for point in discontinuities_y:
        if ys.min() < point < ys.max():
            row = int(np.argmin(np.abs(ys[:, 0] - point)))
            broken[row, :] = np.nan

    return broken


def compose_unary(*functions: NumberFunction, name: str | None = None) -> NumberFunction:
    """Compose unary functions as f1(f2(...fk(x)))."""

    if not functions:
        raise ValueError("At least one unary function is required.")

    def composed(x: float) -> float:
        result = x
        for function in reversed(functions):
            result = function(result)
        return result

    composed.__name__ = name or "composed_unary"
    return composed


def compose_binary_with_unary_output(
    binary_function: BinaryFunction,
    unary_function: NumberFunction,
    *,
    name: str | None = None,
) -> BinaryFunction:
    """Create h(x,y)=unary_function(binary_function(x,y))."""

    def composed(x: float, y: float) -> float:
        return unary_function(binary_function(x, y))

    composed.__name__ = name or "composed_binary_output"
    return composed


def compose_unary_inputs_with_binary(
    binary_function: BinaryFunction,
    x_function: NumberFunction,
    y_function: NumberFunction | None = None,
    *,
    name: str | None = None,
) -> BinaryFunction:
    """Create h(x,y)=binary_function(x_function(x), y_function(y))."""

    y_function = x_function if y_function is None else y_function

    def composed(x: float, y: float) -> float:
        return binary_function(x_function(x), y_function(y))

    composed.__name__ = name or "composed_binary_inputs"
    return composed


def plot_unary_operator_2d(
    function: NumberFunction,
    *,
    domain: tuple[float, float] = (0.0, 1.0),
    points: int = 400,
    title: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    discontinuities: Sequence[float] = (0.0,),
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot one unary operator, e.g. fuzzy negation or supplementary operator."""

    xs = np.linspace(domain[0], domain[1], points)
    ys = _safe_unary_values(function, xs)
    xs, ys = _break_at_discontinuities(xs, ys, discontinuities)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
    ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or _function_name(function, "Unary operator"))
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_unary_operators_2d(
    functions: Sequence[NumberFunction],
    *,
    labels: Sequence[str] | None = None,
    domain: tuple[float, float] = (0.0, 1.0),
    points: int = 400,
    title: str = "Unary operators",
    x_label: str = "x",
    y_label: str = "y",
    discontinuities: Sequence[float] = (0.0,),
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot several unary operators on one 2D chart."""

    if labels is not None and len(labels) != len(functions):
        raise ValueError("labels must have the same length as functions.")

    xs = np.linspace(domain[0], domain[1], points)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for index, function in enumerate(functions):
        label = labels[index] if labels is not None else _function_name(function, f"f{index + 1}")
        plot_xs, plot_ys = _break_at_discontinuities(
            xs,
            _safe_unary_values(function, xs),
            discontinuities,
        )
        ax.plot(plot_xs, plot_ys, linewidth=2, label=label)

    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
    ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_negation(
    negation: NumberFunction,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot fuzzy negation n:[0,1]->[0,1]."""

    return plot_unary_operator_2d(
        negation,
        domain=(0.0, 1.0),
        title=_function_name(negation, "Fuzzy negation"),
        x_label="x",
        y_label="n(x)",
        discontinuities=(),
        save_path=save_path,
        show=show,
    )


def plot_supplementary_operator(
    operator: NumberFunction,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot supplementary operator I:[-1,1]->[-1,1]."""

    return plot_unary_operator_2d(
        operator,
        domain=(-1.0, 1.0),
        title=_function_name(operator, "Supplementary operator"),
        x_label="x",
        y_label="I(x)",
        discontinuities=(0.0,),
        save_path=save_path,
        show=show,
    )


def plot_binary_operator_3d(
    function: BinaryFunction,
    *,
    domain_x: tuple[float, float] = (0.0, 1.0),
    domain_y: tuple[float, float] = (0.0, 1.0),
    points: int = 80,
    title: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    discontinuities_x: Sequence[float] = (),
    discontinuities_y: Sequence[float] = (),
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot a binary operator as a 3D surface."""

    x_values = np.linspace(domain_x[0], domain_x[1], points)
    y_values = np.linspace(domain_y[0], domain_y[1], points)
    xs, ys = np.meshgrid(x_values, y_values)
    zs = _safe_binary_values(function, xs, ys)
    zs = _break_binary_at_discontinuities(
        xs,
        ys,
        zs,
        discontinuities_x=discontinuities_x,
        discontinuities_y=discontinuities_y,
    )

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes)
    ax.plot_surface(xs, ys, zs, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title or _function_name(function, "Binary operator"))
    return _finalize_figure(fig, save_path=save_path, show=show)


def plot_t_norm(
    t_norm: BinaryFunction,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot t-norm T:[0,1]^2->[0,1]."""

    return plot_binary_operator_3d(
        t_norm,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        title=_function_name(t_norm, "T-norm"),
        z_label="T(x,y)",
        discontinuities_x=(),
        discontinuities_y=(),
        save_path=save_path,
        show=show,
    )


def plot_t_conorm(
    t_conorm: BinaryFunction,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot t-conorm S:[0,1]^2->[0,1]."""

    return plot_binary_operator_3d(
        t_conorm,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        title=_function_name(t_conorm, "T-conorm"),
        z_label="S(x,y)",
        discontinuities_x=(),
        discontinuities_y=(),
        save_path=save_path,
        show=show,
    )


def plot_balanced_binary_operator(
    operator: BinaryFunction,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot balanced binary operator on [-1,1]^2."""

    return plot_binary_operator_3d(
        operator,
        domain_x=(-1.0, 1.0),
        domain_y=(-1.0, 1.0),
        title=_function_name(operator, "Balanced binary operator"),
        z_label="P(x,y)",
        discontinuities_x=(0.0,),
        discontinuities_y=(0.0,),
        save_path=save_path,
        show=show,
    )


def _rule_tree_layout(
    trace: RuleEvaluationTrace,
) -> tuple[
    list[tuple[RuleEvaluationTrace, float, float]],
    list[tuple[float, float, float, float]],
    int,
    int,
]:
    """Return positioned trace nodes, edges, leaf count, and maximum depth."""

    positioned = []
    edges = []
    next_leaf_x = 0
    maximum_depth = 0

    def position_node(node_trace: RuleEvaluationTrace, depth: int) -> float:
        nonlocal next_leaf_x, maximum_depth
        maximum_depth = max(maximum_depth, depth)

        child_positions = [
            position_node(child_trace, depth + 1)
            for child_trace in node_trace.children
        ]
        if child_positions:
            x = sum(child_positions) / len(child_positions)
        else:
            x = float(next_leaf_x)
            next_leaf_x += 1

        y = -float(depth)
        positioned.append((node_trace, x, y))
        for child_x in child_positions:
            edges.append((x, y, child_x, y - 1.0))
        return x

    position_node(trace, 0)
    return positioned, edges, max(next_leaf_x, 1), maximum_depth


def plot_rule_tree(
    rule_node: RuleNode,
    variables: Mapping[str, float],
    operators: Mapping[str, OperatorFunction],
    *,
    show_values: bool = True,
    title: str = "Rule evaluation tree",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot a rule tree using values produced by ``evaluate_with_trace``."""

    _, trace = rule_node.evaluate_with_trace(variables, operators)
    positioned, edges, leaf_count, maximum_depth = _rule_tree_layout(trace)

    width = max(7.0, leaf_count * 1.7)
    height = max(4.5, (maximum_depth + 1) * 1.45)
    fig, ax = plt.subplots(figsize=(width, height))

    for parent_x, parent_y, child_x, child_y in edges:
        ax.plot(
            [parent_x, child_x],
            [parent_y, child_y],
            color="#6b7280",
            linewidth=1.4,
            zorder=1,
        )

    for node_trace, x, y in positioned:
        label = node_trace.node.symbol
        if show_values:
            label = f"{label}\n= {node_trace.value:.6g}"

        is_variable = node_trace.node.is_variable
        ax.text(
            x,
            y,
            label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            fontweight="bold" if not is_variable else "normal",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#dcfce7" if is_variable else "#dbeafe",
                "edgecolor": "#15803d" if is_variable else "#1d4ed8",
                "linewidth": 1.4,
            },
            zorder=2,
        )

    x_padding = 0.75 if leaf_count > 1 else 1.0
    ax.set_xlim(-x_padding, max(leaf_count - 1, 0) + x_padding)
    ax.set_ylim(-maximum_depth - 0.75, 0.75)
    ax.set_title(title)
    ax.axis("off")
    return _finalize_figure(fig, save_path=save_path, show=show)


def save_many_unary_plots(
    functions: Iterable[NumberFunction],
    output_dir: str | Path,
    *,
    domain: tuple[float, float],
    points: int = 400,
) -> list[Path]:
    """Save one 2D plot for each unary operator and return created paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, function in enumerate(functions, start=1):
        name = _function_name(function, f"operator_{index}").replace(" ", "_")
        path = output / f"{index:02d}_{name}.png"
        fig = plot_unary_operator_2d(function, domain=domain, points=points, save_path=path)
        plt.close(fig)
        paths.append(path)
    return paths
