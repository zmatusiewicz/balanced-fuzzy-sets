# Balanced Fuzzy Operators Toolkit

This project is a small research-oriented Python toolkit for constructing,
checking, composing, evaluating, and visualizing fuzzy operators on two domains:

- the classical fuzzy interval `[0,1]`,
- the balanced interval `[-1,1]`.

It includes classical fuzzy negations, t-norms, t-conorms, additive generators,
representable uninorms, nullnorms, balanced fuzzy negations, balanced t-norms,
balanced t-conorms, supplementary operators, compact prefix-notation rules, and
a Petri-inspired layered transition graph for staged analysis.

The code distinguishes mathematical guarantees from numerical checks. Finite
grid checks are useful regression tests, but they are not treated as proofs on a
continuous interval.

## Installation

Python 3.10 through 3.14 is supported.

Create or activate a Python environment, then install the package:

```bash
pip install -e .
```

The current requirements are:

- `numpy`
- `matplotlib`

Development tools, including the test runner, can be installed with:

```bash
pip install -e ".[dev]"
```

The most commonly used classes and operations are available from the public
package API. Factory families with overlapping names are grouped in submodules:

```python
from balanced_fuzzy_sets import BalancedFuzzyOperators, build_rule
from balanced_fuzzy_sets import fuzzy_negations, tnorm_additive_generators
```

Run the complete example:

```bash
python examples/main_example.py
```

The demonstration reads its input from `examples/data` and writes plots, CSV
tables, and transition summaries to `examples/generated`.

## API Documentation

The API reference is generated from the package docstrings with Sphinx. Build
it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to browse the generated documentation. The
source files are in `docs`, starting with `docs/api.rst`.

The repository includes `.readthedocs.yaml` for hosted builds. After importing
the GitHub repository into Read the Docs, the service installs the package with
the `docs` extra, builds `docs/conf.py`, and rejects builds containing Sphinx
warnings.

## Core Concepts

### Guarantee Metadata

Main operator objects expose a `guarantee` field:

- `by construction` - the property follows from a known family or theorem-level
  construction,
- `symbolically verified` - the property was verified symbolically,
- `numerically sampled` - the property was checked only on a finite grid,
- `user supplied, unchecked` - the function was supplied without validation.

Numerical grid checks are stored in `numerically_checked_properties`. The old
name `verified_properties` remains only as a backward-compatible alias.

### Balanced Fuzzy Negation

A basic balanced fuzzy negation is treated as a function
`N_B:[-1,1]->[-1,1]` satisfying:

- mapping into `[-1,1]`,
- decreasing behavior,
- `N_B(0)=0`,
- sign reversal condition `x*N_B(x)<=0`.

Strictness and strength are separated:

- strict balanced fuzzy negation: basic balanced fuzzy negation plus bijectivity,
- strong balanced fuzzy negation: basic balanced fuzzy negation plus involutivity.

The implementation does not infer surjectivity from distinct sampled values.
The relevant metadata is:

- `empirically_injective_on_grid`: sampled values do not repeat on the
  validation grid,
- `surjectivity_status`: `guaranteed_by_continuity`, `failed_endpoints`, or
  `unknown`,
- `empirically_bijective_on_grid`: `True` only when surjectivity is guaranteed
  by continuity, strict decrease is observed on the grid, and endpoint values
  are correct,
- `empirically_strict_on_grid`: base balanced fuzzy negation plus confirmed
  bijectivity under the metadata above.

Endpoint behavior is checked as part of bijectivity, not as part of the base
balanced-negation definition. For arbitrary functions with jumps, surjectivity
cannot be confirmed by grid sampling, so the metadata is `unknown`, not `True`.

## Building Balanced Fuzzy Operators

### Balanced Negation From a Classical Fuzzy Negation

```python
from balanced_fuzzy_sets import BalancedFuzzyOperators
from balanced_fuzzy_sets.fuzzy_negations import cosine

fuzzy_negation = cosine()

balanced_negation = (
    BalancedFuzzyOperators.create_balanced_negation_from_one_fuzzy_negation(
        fuzzy_negation,
        name="Cosine balanced negation",
        formula=(
            "N_B(x)=n_cos(x)-1 for x>0; "
            "N_B(x)=1-n_cos(-x) for x<0; N_B(0)=0"
        ),
        is_strict=True,
        is_strong=False,
        is_continuous=True,
    )
)

print(balanced_negation(0.4))
print(balanced_negation.numerically_checked_properties.is_balanced_negation)
print(balanced_negation.guarantee)
```

### Balanced T-Norm From a Classical T-Norm

```python
from balanced_fuzzy_sets import BalancedFuzzyOperators
from balanced_fuzzy_sets.tnorm_additive_generators import aczel_alsina

generator = aczel_alsina(3.0)
balanced_t_norm = BalancedFuzzyOperators.balanced_t_norm_from_t_norm(
    generator.t_norm
)

print(balanced_t_norm(0.2, 0.3))
print(balanced_t_norm(-0.2, -0.3))
```

The dual t-conorm is generated internally as:

```text
S(x,y)=1-T(1-x,1-y)
```

If a t-conorm is supplied explicitly, use:

```python
BalancedFuzzyOperators.balanced_t_norm_from_dual_pair(t_norm, t_conorm)
```

The pair is rejected when the numerical duality check fails.

The resulting balanced t-norm is also checked on a finite grid for the core
properties required by the implemented balanced definition:

- mapping `[-1,1]^2` into `[-1,1]`,
- commutativity,
- associativity,
- monotonicity,
- boundary behavior,
- balanced symmetry.

The boundary convention used by this construction is:

```text
P(x, 1)=P(1, x)=max(x, 0)
P(x,-1)=P(-1,x)=min(x, 0)
```

If any empirical check fails, the constructor raises `ValueError` instead of
returning a callable labelled as a balanced t-norm.

### Balanced T-Conorm From a Representable Uninorm

```python
from balanced_fuzzy_sets import (
    BalancedFuzzyOperators,
    additive_generator_of_representable_uninorm,
)
from balanced_fuzzy_sets.tnorm_additive_generators import aczel_alsina

generator = aczel_alsina(3.0)
uninorm_generator = additive_generator_of_representable_uninorm(generator, e=0.5)

f = lambda x: (x + 1.0) / 2.0
f_inverse = lambda u: 2.0 * u - 1.0

balanced_t_conorm = BalancedFuzzyOperators.create_balanced_t_conorm(
    uninorm_generator,
    f,
    f_inverse,
    opposite_corner_convention="disjunctive",
)

print(balanced_t_conorm(0.4, -0.2))
```

The transport requires the neutral element to be mapped to `0`. With the
standard map `f(x)=(x+1)/2`, this forces `e=0.5`.

For the balanced t-conorm construction, the validator also checks both
symmetry assumptions required by the theorem:

```text
f(-x)=n(f(x))
u(n(x))=-u(x)
```

The second condition is checked on the unit grid used by the validator and is
stored as `empirically_uninorm_generator_antisymmetric_on_grid` in the returned
callable metadata.

### Balanced T-Norm From a Nullnorm

```python
from balanced_fuzzy_sets import (
    BalancedFuzzyOperators,
    nullnorm_from_t_norm_and_t_conorm,
)
from balanced_fuzzy_sets.tnorm_additive_generators import aczel_alsina

generator = aczel_alsina(3.0)
nullnorm = nullnorm_from_t_norm_and_t_conorm(generator.t_norm, z=0.5)

f = lambda x: (x + 1.0) / 2.0
f_inverse = lambda u: 2.0 * u - 1.0

balanced_t_norm = BalancedFuzzyOperators.balanced_t_norm_from_nullnorm(
    nullnorm,
    f,
    f_inverse,
)
```

The general nullnorm transport is also available through `transport_nullnorm`,
but it does not claim balanced symmetry. `balanced_t_norm_from_nullnorm`
performs the full balanced t-norm grid check after transport.

## Evaluating Composite Fuzzy Expressions

Demonstration rules are stored in compact prefix notation in
`examples/data/rules`. Operator symbols are:

- `N` - unary balanced fuzzy negation,
- `R` - unary supplementary operator,
- `T` - binary t-norm,
- `S` - binary t-conorm.

For example:

```text
STxNxRx
```

is parsed as:

```text
S(T(x, N(x)), R(x))
```

Build and evaluate a rule:

```python
from balanced_fuzzy_sets import (
    BalancedFuzzyOperators,
    additive_generator_of_representable_uninorm,
    build_rule,
    rule_to_infix,
)
from balanced_fuzzy_sets.fuzzy_negations import cosine
from balanced_fuzzy_sets.supplementary_operators import cosine_supplementary_operator
from balanced_fuzzy_sets.tnorm_additive_generators import aczel_alsina

generator = aczel_alsina(3.0)
uninorm_generator = additive_generator_of_representable_uninorm(generator, e=0.5)
f = lambda x: (x + 1.0) / 2.0
f_inverse = lambda u: 2.0 * u - 1.0

balanced_t_norm = BalancedFuzzyOperators.balanced_t_norm_from_t_norm(
    generator.t_norm
)
balanced_t_conorm = BalancedFuzzyOperators.create_balanced_t_conorm(
    uninorm_generator,
    f,
    f_inverse,
)
balanced_negation = (
    BalancedFuzzyOperators.create_balanced_negation_from_one_fuzzy_negation(
        cosine(),
        is_strict=True,
        is_strong=False,
        is_continuous=True,
    )
)
supplementary = cosine_supplementary_operator()

rule_code = "STxNxRx"
rule = build_rule(
    rule_code,
    {
        "T": balanced_t_norm,
        "S": balanced_t_conorm,
        "N": balanced_negation,
        "R": supplementary,
    },
)

print(rule_to_infix(rule_code))
print(rule(x=0.4))
```

The parser rejects unknown uppercase symbols and rejects calls with extra or
missing variables. This avoids silently accepting malformed expressions.

## Saving Rule Value Tables

`examples/main_example.py` includes `save_rule_table(...)`, which evaluates a
two-variable rule for `x,y` from `-1` to `1` with a chosen step:

```python
from pathlib import Path

save_rule_table(rule_surface, Path("examples/generated/rule_values.csv"), step=0.1)
```

The output CSV has columns:

```text
x,y,rule_value
```

## Visualizing Operators and Rules

The `balanced_fuzzy_sets.graphical_representation` module contains plotting helpers:

A detailed manual is available in [`docs/manual.rst`](docs/manual.rst).

- `plot_unary_operator_2d(...)` - for negations and other unary functions,
- `plot_supplementary_operator(...)` - for supplementary operators,
- `plot_binary_operator_3d(...)` - for classical binary operators on `[0,1]`,
- `plot_balanced_binary_operator(...)` - for binary operators on `[-1,1]`.

Example:

```python
from balanced_fuzzy_sets import (
    plot_balanced_binary_operator,
    plot_supplementary_operator,
    plot_unary_operator_2d,
)

plot_balanced_binary_operator(
    balanced_t_norm,
    save_path="examples/generated/balanced_t_norm.png",
)

plot_unary_operator_2d(
    balanced_negation,
    domain=(-1.0, 1.0),
    title=balanced_negation.name,
    y_label="N_B(x)",
    save_path="examples/generated/balanced_negation.png",
)

plot_supplementary_operator(
    supplementary,
    save_path="examples/generated/supplementary_operator.png",
)
```

The plotting code avoids drawing lines through detected discontinuity points.

## Visualizing an Analysis Process

Demonstration process steps are stored in `examples/data/process_steps`. A step
file has this format:

```text
Stage1. x1=0.5, x2=0.7, x3=-0.2; 1: x1:=Rule1(x1); 2: x2:=Rule2(x2,x1,x3)
```

Each rule call includes:

- priority,
- explicit output place,
- rule name,
- input places.

The output place is updated by the fired rule. In process mode, the first file
initializes the state and later step files must agree with the state produced by
previous steps.

Build and draw a Petri-inspired transition graph:

```python
from pathlib import Path
from balanced_fuzzy_sets import (
    build_petri_inspired_network_from_process_steps,
    draw_petri_inspired_network,
    petri_inspired_network_summary,
)

network = build_petri_inspired_network_from_process_steps(
    [
        Path("examples/data/process_steps") / "Step1",
        Path("examples/data/process_steps") / "Step2",
        Path("examples/data/process_steps") / "Step3",
    ],
    {
        "T": balanced_t_norm,
        "S": balanced_t_conorm,
        "N": balanced_negation,
        "R": supplementary,
    },
    rules_dir=Path("examples/data/rules"),
    stage_name="Three-step Petri-inspired transition graph",
)

draw_petri_inspired_network(
    network,
    save_path="examples/generated/petri_inspired_3_steps.png",
)

print(petri_inspired_network_summary(network))
```

The graph is layered:

```text
xk(1) -> rule transition -> xk(2) -> next rule transition -> xk(3)
```

Places are drawn as circles. Rule applications are transitions. There are no
direct place-to-place edges.

## Module Reference

### `balanced_fuzzy_sets.balanced_fuzzy_operators`

Factory class for balanced operators. It contains constructors for:

- balanced fuzzy negations from one fuzzy negation,
- balanced fuzzy negations from two fuzzy negations,
- balanced fuzzy negations from normalized bijections,
- balanced fuzzy negations from scaled bijections,
- balanced fuzzy negations from odd/even generator patterns,
- balanced t-norms from a classical t-norm,
- balanced t-norms from a verified dual pair,
- balanced t-conorms from transported representable uninorms,
- general nullnorm transports,
- balanced t-norms from nullnorms.

It performs domain checks, inverse checks, duality checks, theorem-level
transport checks such as `u(n(x))=-u(x)`, and full balanced t-norm grid checks
where appropriate. A callable is not accepted as a balanced operator merely
because it has the right Python signature.

### `balanced_fuzzy_sets.balanced_fuzzy_negations`

Defines:

- `BalancedFuzzyNegation`,
- `BalancedNegationProperties`,
- known table examples: reversal, square, cube, square root, cosine, Sugeno,
- summary and factory lookup helpers.

The class separates base balanced negation, strictness, and strength.

### `balanced_fuzzy_sets.fuzzy_negations`

Defines:

- `FuzzyNegation`,
- `FuzzyNegationProperties`,
- standard/Zadeh negation,
- Yager negation,
- Sugeno negation,
- power and root families,
- cosine negation,
- exponential generated negation,
- Dombi/Sugeno fixed-point negation,
- generated strong negation constructor,
- intuitionistic and greatest pointwise negations.

### `balanced_fuzzy_sets.fuzzy_norms`

Defines:

- `TNorm`,
- `TConorm`,
- `FuzzyNormProperties`,
- basic and parametric t-norm families,
- dual construction methods,
- factory dictionaries for known t-norms and t-conorms.

### `balanced_fuzzy_sets.tnorm_additive_generators`

Defines additive generators and inverse or pseudo-inverse functions for:

- Lukasiewicz,
- product,
- drastic,
- Schweizer-Sklar,
- Hamacher,
- Frank,
- Yager,
- Aczel-Alsina,
- Dombi,
- Sugeno-Weber.

The module uses stable numerical forms such as `expm1` and `log1p` for sensitive
parameter ranges.

### `balanced_fuzzy_sets.representable_uninorm_generator`

Defines:

- `AdditiveGeneratorOfRepresentableUninorm`,
- `additive_generator_of_representable_uninorm(...)`.

It constructs a representable uninorm generator from a strict t-norm additive
generator and neutral element `e`.

### `balanced_fuzzy_sets.nullnorms`

Defines:

- `Nullnorm`,
- `NullnormProperties`,
- `nullnorm_from_t_norm_and_t_conorm(...)`,
- `transport_nullnorm(...)`,
- `balanced_t_norm_from_nullnorm(...)`.

### `balanced_fuzzy_sets.supplementary_operators`

Defines:

- `SupplementaryOperator`,
- a generic supplementary operator induced by a fuzzy negation,
- known supplementary operator factories based on standard, Yager, Sugeno,
  power, root, and cosine negations.

### `balanced_fuzzy_sets.rule_builder`

Defines a parser and evaluator for compact prefix-notation rules. It provides:

- syntax tree nodes,
- parser entry points,
- infix formatting,
- callable rule construction,
- strict variable and operator validation.

### `balanced_fuzzy_sets.transition_analysis`

Defines:

- `RuleCall`,
- `AnalysisStage`,
- `RuleResult`,
- stage parsing,
- stage evaluation,
- multi-stage evaluation,
- result formatting.

### `balanced_fuzzy_sets.petri_inspired_network`

Defines:

- `Place`,
- `Transition`,
- `Arc`,
- `PetriInspiredNetwork`,
- builders for one stage, process steps, and independent scenarios,
- drawing functions,
- text summary functions.

### `balanced_fuzzy_sets.graphical_representation`

Defines plotting functions for unary and binary fuzzy operators, including
balanced-domain plots and discontinuity-aware rendering.

### `balanced_fuzzy_sets.operator_guarantees`

Defines guarantee labels and helpers used by operator objects:

- `BY_CONSTRUCTION`,
- `SYMBOLICALLY_VERIFIED`,
- `NUMERICALLY_SAMPLED`,
- `USER_SUPPLIED_UNCHECKED`,
- `normalize_guarantee(...)`,
- `attach_guarantee(...)`.

### `examples/main_example.py`

Executable example that combines the project features:

- builds an Aczel-Alsina generator with `p=3`,
- constructs a representable uninorm,
- constructs balanced t-norms and t-conorms,
- constructs a balanced negation and supplementary operator,
- loads and evaluates `Rule1`,
- plots operators and rules,
- saves a CSV value table,
- builds and draws a three-step Petri-inspired transition graph.

## Project Directories

- `examples/data/rules` - demonstration rule files `Rule1` through `Rule10`,
- `examples/data/process_steps` - demonstration process files `Step1` through
  `Step4`,
- `examples/data/transition_scenarios` - independent demonstration scenarios,
- `examples/generated` - generated plots, CSV files, and text summaries.
