import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from balanced_fuzzy_sets import (
    RuleEvaluationTrace,
    build_rule,
    parse_rule,
    plot_rule_tree,
)


def _operators():
    return {
        "T": lambda x, y: x * y,
        "S": lambda x, y: x + y,
        "N": lambda x: 1.0 - x,
        "R": lambda x: -x,
    }


def _walk_trace(trace: RuleEvaluationTrace):
    yield trace
    for child in trace.children:
        yield from _walk_trace(child)


def test_parser_recognizes_rule_tree():
    rule_node = parse_rule("STxNxRx")

    assert rule_node.symbol == "S"
    assert [child.symbol for child in rule_node.children] == ["T", "R"]
    assert rule_node.to_infix() == "S(T(x, N(x)), R(x))"
    assert rule_node.variables() == {"x"}


def test_evaluate_with_trace_preserves_the_complete_rule_tree():
    rule_node = parse_rule("STxNxRx")

    result, trace = rule_node.evaluate_with_trace({"x": 0.4}, _operators())

    assert result == pytest.approx(-0.16)
    assert trace.node is rule_node
    assert trace.value == pytest.approx(result)
    t_trace, r_trace = trace.children
    assert t_trace.value == pytest.approx(0.24)
    assert r_trace.value == pytest.approx(-0.4)
    assert t_trace.children[1].node.symbol == "N"
    assert t_trace.children[1].value == pytest.approx(0.6)
    assert [item.node.symbol for item in _walk_trace(trace)] == [
        "S",
        "T",
        "x",
        "N",
        "x",
        "R",
        "x",
    ]

    x_traces = [item for item in _walk_trace(trace) if item.node.symbol == "x"]
    assert len(x_traces) == 3
    assert len({id(item) for item in x_traces}) == 3
    assert all(item.value == pytest.approx(0.4) for item in x_traces)


def test_rule_callable_and_evaluate_with_trace_have_identical_results():
    rule_node = parse_rule("STxNxRx")
    variables = {"x": 0.4}
    operators = _operators()
    rule = build_rule("STxNxRx", operators)

    normal_result = rule(**variables)
    traced_result, trace = rule_node.evaluate_with_trace(variables, operators)

    assert traced_result == pytest.approx(normal_result)
    assert trace.value == pytest.approx(normal_result)
    assert trace.node.symbol == "S"


def test_evaluate_with_trace_preserves_evaluation_errors():
    rule_node = parse_rule("TxNx")

    with pytest.raises(KeyError, match="Missing operator implementation for 'T'"):
        rule_node.evaluate_with_trace({"x": 0.4}, {})

    with pytest.raises(KeyError, match="x"):
        rule_node.evaluate_with_trace({}, _operators())


def test_plot_rule_tree_renders_every_occurrence_and_saves_the_figure(tmp_path):
    rule_node = parse_rule("STxNxRx")
    output_path = tmp_path / "rule_tree.png"

    figure = plot_rule_tree(
        rule_node,
        {"x": 0.4},
        _operators(),
        save_path=output_path,
    )
    try:
        labels = [text.get_text() for text in figure.axes[0].texts]

        assert output_path.is_file()
        assert len(labels) == 7
        assert labels.count("x\n= 0.4") == 3
        assert "S\n= -0.16" in labels
        assert len(figure.axes[0].lines) == 6
    finally:
        plt.close(figure)


def test_plot_rule_tree_evaluates_each_operator_once():
    rule_node = parse_rule("STxNxRx")
    calls = {symbol: 0 for symbol in "TSNR"}

    def counted(symbol, function):
        def operator(*values):
            calls[symbol] += 1
            return function(*values)

        return operator

    operators = {
        "T": counted("T", lambda x, y: x * y),
        "S": counted("S", lambda x, y: x + y),
        "N": counted("N", lambda x: 1.0 - x),
        "R": counted("R", lambda x: -x),
    }

    figure = plot_rule_tree(rule_node, {"x": 0.4}, operators)
    try:
        assert calls == {"T": 1, "S": 1, "N": 1, "R": 1}
    finally:
        plt.close(figure)


def test_plot_rule_tree_can_hide_values():
    figure = plot_rule_tree(
        parse_rule("TxNx"),
        {"x": 0.4},
        _operators(),
        show_values=False,
    )
    try:
        labels = [text.get_text() for text in figure.axes[0].texts]
        assert labels == ["x", "x", "N", "T"]
    finally:
        plt.close(figure)


def test_plot_rule_tree_supports_a_single_variable_rule():
    figure = plot_rule_tree(parse_rule("x"), {"x": 0.25}, {})
    try:
        axes = figure.axes[0]
        assert [text.get_text() for text in axes.texts] == ["x\n= 0.25"]
        assert len(axes.lines) == 0
    finally:
        plt.close(figure)
