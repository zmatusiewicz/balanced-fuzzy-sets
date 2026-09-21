"""Run the complete artifact-generating example from the source tree."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from balanced_fuzzy_sets import (
    BalancedFuzzyOperators,
    additive_generator_of_representable_uninorm,
    build_petri_inspired_network_from_process_steps,
    build_rule,
    draw_petri_inspired_network,
    nullnorm_from_t_norm_and_t_conorm,
    nullnorms_summary,
    parse_rule,
    petri_inspired_network_summary,
    plot_balanced_binary_operator,
    plot_binary_operator_3d,
    plot_rule_tree,
    plot_supplementary_operator,
    plot_unary_operator_2d,
    rule_to_infix,
)
from balanced_fuzzy_sets.fuzzy_negations import (
    cosine,
    dombi_with_fixed_point,
    negations_summary,
    power,
    yager,
)
from balanced_fuzzy_sets.fuzzy_norms import fuzzy_norms_summary
from balanced_fuzzy_sets.supplementary_operators import cosine_supplementary_operator
from balanced_fuzzy_sets.tnorm_additive_generators import (
    aczel_alsina,
    literature_summary,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
RULES_DIR = DATA_DIR / "rules"
OUTPUT_DIR = EXAMPLES_DIR / "generated"
RULE_PATH = RULES_DIR / "Rule1"
RULE_TABLE_PATH = OUTPUT_DIR / "rule_values.csv"
PETRI_INSPIRED_SUMMARY_PATH = OUTPUT_DIR / "petri_inspired_3_steps_summary.txt"


def save_rule_table(rule_function, path: Path, *, step: float = 0.1) -> None:
    """Save values of a two-variable rule for x,y in [-1,1]."""

    values = [round(-1.0 + index * step, 10) for index in range(int(2.0 / step) + 1)]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "rule_value"])
        for x in values:
            for y in values:
                writer.writerow([f"{x:.1f}", f"{y:.1f}", f"{rule_function(x, y):.12f}"])


if __name__ == "__main__":
    print(literature_summary())
    print()
    print(fuzzy_norms_summary())
    print()
    g = aczel_alsina(3.0)
    print(f"Example: {g.name}, {g.formula}")
    print(f"T(0.7, 0.8) = {g.t_norm(0.7, 0.8):.12f}")
    print()
    u = additive_generator_of_representable_uninorm(g, e=0.5)
    print(f"Example: {u.name}")
    print(f"U(0.3, 0.8) = {u.uninorm(0.3, 0.8):.12f}")
    print()
    print(negations_summary())
    print()
    print(nullnorms_summary())
    print()
    for n in [yager(2.0), power(2.0), dombi_with_fixed_point(0.4)]:
        print(f"{n.name}: n(0.6) = {n(0.6):.12f}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    selected_generator = aczel_alsina(3.0)
    selected_uninorm_generator = additive_generator_of_representable_uninorm(
        selected_generator,
        e=0.5,
    )

    f = lambda x: (x + 1.0) / 2.0
    f_inverse = lambda x: 2.0 * x - 1.0

    balanced_t_norm = BalancedFuzzyOperators.balanced_t_norm_from_t_norm(
        selected_generator.t_norm
    )
    balanced_t_conorm_one = BalancedFuzzyOperators.create_balanced_t_conorm(
        selected_uninorm_generator,
        f,
        f_inverse,
        opposite_corner_convention="disjunctive",
    )
    balanced_t_conorm_minus_one = BalancedFuzzyOperators.create_balanced_t_conorm(
        selected_uninorm_generator,
        f,
        f_inverse,
        opposite_corner_convention="conjunctive",
    )
    nullnorm = nullnorm_from_t_norm_and_t_conorm(
        selected_generator.t_norm,
        z=0.5,
        name="Aczel-Alsina generated nullnorm(z=0.5)",
    )
    balanced_t_norm_from_nullnorm = BalancedFuzzyOperators.balanced_t_norm_from_nullnorm(
        nullnorm,
        f,
        f_inverse,
    )

    fuzzy_negation = cosine()
    balanced_negation = BalancedFuzzyOperators.create_balanced_negation_from_one_fuzzy_negation(
        fuzzy_negation,
        name="Cosine balanced negation",
        formula="N_B(x)=n_cos(x)-1 for x>0; N_B(x)=1-n_cos(-x) for x<0; N_B(0)=0",
        is_strict=True,
        is_strong=False,
        is_continuous=True,
    )
    supplementary = cosine_supplementary_operator()

    rule_code = RULE_PATH.read_text(encoding="utf-8").strip()
    rule_node = parse_rule(rule_code)
    rule_operators = {
        "T": balanced_t_norm,
        "S": balanced_t_conorm_one,
        "N": balanced_negation,
        "R": supplementary,
    }
    example_values = {"x": 0.2}
    rule = build_rule(rule_code, rule_operators)
    rule_variables = rule_node.variables()
    rule_surface = lambda x, y: rule(
        **{
            variable: {"x": x, "y": y}[variable]
            for variable in rule_variables
        }
    )

    figures = [
        plot_binary_operator_3d(
            selected_uninorm_generator.uninorm,
            save_path=OUTPUT_DIR / "aczel_alsina_p3_uninorm.png",
        ),
        plot_balanced_binary_operator(
            balanced_t_norm,
            save_path=OUTPUT_DIR / "aczel_alsina_p3_balanced_t_norm.png",
        ),
        plot_balanced_binary_operator(
            balanced_t_conorm_one,
            save_path=OUTPUT_DIR / "aczel_alsina_p3_balanced_t_conorm_corner_1.png",
        ),
        plot_balanced_binary_operator(
            balanced_t_conorm_minus_one,
            save_path=OUTPUT_DIR / "aczel_alsina_p3_balanced_t_conorm_corner_minus_1.png",
        ),
        plot_balanced_binary_operator(
            balanced_t_norm_from_nullnorm,
            save_path=OUTPUT_DIR / "aczel_alsina_p3_balanced_t_norm_from_nullnorm.png",
        ),
        plot_unary_operator_2d(
            balanced_negation,
            domain=(-1.0, 1.0),
            title=balanced_negation.name,
            y_label="N_B(x)",
            save_path=OUTPUT_DIR / "cosine_balanced_negation.png",
        ),
        plot_supplementary_operator(
            supplementary,
            save_path=OUTPUT_DIR / "cosine_supplementary_operator.png",
        ),
        plot_balanced_binary_operator(
            rule_surface,
            save_path=OUTPUT_DIR / "rule_Rule1.png",
        ),
        plot_rule_tree(
            rule_node,
            example_values,
            rule_operators,
            show_values=True,
            save_path=OUTPUT_DIR / "rule_Rule1_tree.png",
        ),
    ]
    for figure in figures:
        plt.close(figure)

    save_rule_table(rule_surface, RULE_TABLE_PATH)

    petri_inspired_network = build_petri_inspired_network_from_process_steps(
        [
            DATA_DIR / "process_steps" / "Step1",
            DATA_DIR / "process_steps" / "Step2",
            DATA_DIR / "process_steps" / "Step3",
        ],
        rule_operators,
        rules_dir=RULES_DIR,
        stage_name="Three-step Petri-inspired transition graph",
    )
    petri_inspired_summary = petri_inspired_network_summary(petri_inspired_network)
    PETRI_INSPIRED_SUMMARY_PATH.write_text(petri_inspired_summary, encoding="utf-8")
    petri_inspired_figure = draw_petri_inspired_network(
        petri_inspired_network,
        save_path=OUTPUT_DIR / "petri_inspired_3_steps.png",
    )
    plt.close(petri_inspired_figure)

    print()
    print(f"Rule: {rule_code} = {rule_to_infix(rule_code)}")
    print()
    print(petri_inspired_summary)
    print(f"Saved plots in: {OUTPUT_DIR}")
    print(f"Saved rule table: {RULE_TABLE_PATH}")
    print(f"Saved Petri-inspired network summary: {PETRI_INSPIRED_SUMMARY_PATH}")
