# Project Overview

This project contains modules for building, validating, composing, and plotting
fuzzy operators on the classical interval `[0,1]` and the balanced interval
`[-1,1]`. The example script combines t-norm generators, representable uninorms,
balanced fuzzy negations, nullnorms, supplementary operators, prefix-notation
rules, and a Petri-inspired transition graph for staged analysis.

Each main operator object exposes a `guarantee` field that separates the source
of the mathematical claim from numerical checks:

- `by construction` - the property follows from a known family or construction,
- `symbolically verified` - the property was verified symbolically,
- `numerically sampled` - the property was only checked on a finite grid,
- `user supplied, unchecked` - the function was supplied without validation.

`numerically_checked_properties` contains regression-style checks on finite
grids. It is not a proof of monotonicity, associativity, commutativity, or
involutivity on a continuous interval. The old name `verified_properties`
remains only as a backward-compatible alias.

## `balanced_fuzzy_sets.balanced_fuzzy_operators`

Central factory module for balanced fuzzy operations. It defines
`BalancedFuzzyOperators`, with constructors for balanced fuzzy negations,
balanced t-norms, balanced t-conorms, and nullnorm transports.

Important constructors:

- `create_balanced_negation_from_function(function, ...)`
- `create_balanced_negation_from_one_fuzzy_negation(negation, ...)`
- `create_balanced_negation_from_two_fuzzy_negations(...)`
- `create_balanced_negation_from_bijection(...)`
- `create_balanced_negation_from_scaled_bijection(...)`
- `create_inverse_balanced_negation(...)`
- `create_odd_composition_balanced_negation(...)`
- `create_balanced_negation_from_even_extension(...)`
- `create_balanced_negation_from_odd_generator(...)`
- `create_balanced_negation_from_even_generator(...)`
- `create_balanced_negation_from_positive_generator(...)`
- `balanced_t_norm_from_t_norm(t_norm, ...)`
- `balanced_t_norm_from_dual_pair(t_norm, t_conorm, ...)`
- `create_balanced_t_conorm(uninorm_generator, f, f_inverse, ...)`
- `transport_nullnorm(nullnorm, f, f_inverse, ...)`
- `balanced_t_norm_from_nullnorm(nullnorm, f, f_inverse, ...)`

The balanced t-norm constructor from a dual pair validates the duality condition
`S(x,y)=1-T(1-x,1-y)` on a grid and also checks the balanced symmetry
`P(x,y)=-P(-x,-y)` on a grid. These checks are implementation tests, not
continuous-interval proofs.

The resulting balanced t-norm is checked on a finite grid for mapping into
`[-1,1]`, commutativity, associativity, monotonicity, boundary behavior, and
balanced symmetry. The boundary convention is:

```text
P(x, 1)=P(1, x)=max(x, 0)
P(x,-1)=P(-1,x)=min(x, 0)
```

If any empirical check fails, the constructor raises `ValueError` instead of
returning a callable labelled as a balanced t-norm.

The balanced t-conorm constructor validates the transport map numerically:
`f(-1)=0`, `f(0)=e`, `f(1)=1`, inverse consistency, transport symmetry, and the
neutral element transported to `0`. Opposite corners are controlled by
`opposite_corner_convention`, with values `"disjunctive"` and `"conjunctive"`.
It also checks the theorem-level generator antisymmetry condition on the unit
grid:

```text
u(n(x))=-u(x)
```

The returned callable records this as
`empirically_uninorm_generator_antisymmetric_on_grid`.

## `balanced_fuzzy_sets.balanced_fuzzy_negations`

Defines `BalancedFuzzyNegation` and example balanced fuzzy negations from the
paper table.

The properties are separated as follows:

- `is_balanced_negation`: mapping into `[-1,1]`, decreasing behavior on the
  validation grid, fixed zero, and sign condition `x*N_B(x)<=0`.
- `empirically_injective_on_grid`: sampled values do not repeat on the
  validation grid.
- `surjectivity_status`: `guaranteed_by_continuity`, `failed_endpoints`, or
  `unknown`.
- `empirically_bijective_on_grid`: `True` only when surjectivity is guaranteed
  by continuity, strict decrease is observed on the grid, and endpoint values
  are correct. For arbitrary discontinuous functions it is `None`, not `True`.
- `empirically_strict_on_grid`: basic balanced fuzzy negation plus confirmed
  bijectivity under the metadata above. Endpoint behavior is checked here, not
  in the base definition.
- `empirically_strong_on_grid`: basic balanced fuzzy negation plus empirical
  involutivity on the validation grid.

For user-supplied functions without a theoretical guarantee, declarations such
as `is_strict=True` are stored in `claimed_properties` and are not promoted to a
confirmed property solely from grid sampling. A function with jumps can be
injective on the grid and still skip whole value intervals, so its surjectivity
status remains `unknown`.

## `balanced_fuzzy_sets.fuzzy_negations`

Contains classical fuzzy negations `n:[0,1]->[0,1]`: standard, Yager, Sugeno,
power, complement-power, root-complement, complement-root, cosine, exponential,
Dombi/Sugeno fixed-point, and generated strong negations.

`FuzzyNegation` checks range, boundary conditions `n(0)=1`, `n(1)=0`,
decreasing behavior on a grid, strict decrease on a grid, and involutivity on a
grid. Output values outside `[0,1]` beyond numerical tolerance raise an error
instead of being silently clipped.

## `balanced_fuzzy_sets.fuzzy_norms`

Defines `TNorm`, `TConorm`, and `FuzzyNormProperties`. Implemented families
include minimum/maximum, product/probabilistic sum, Lukasiewicz, drastic,
nilpotent, Hamacher, Einstein, Schweizer-Sklar, Yager, Aczel-Alsina, Dombi,
Frank, Hamacher, and Sugeno-Weber.

Known families and dual constructions are marked as `guarantee="by construction"`.
Arbitrary supplied functions are marked as `numerically sampled` when validation
is enabled or `user supplied, unchecked` when `validate=False`.

## `balanced_fuzzy_sets.supplementary_operators`

Defines supplementary operators `I:[-1,1]->[-1,1]`, usually induced by a
classical fuzzy negation:

```text
I(x)=n(x) for x>0, I(x)=-n(-x) for x<0, I(0)=0
```

## `balanced_fuzzy_sets.tnorm_additive_generators`

Defines additive generators for t-norms and literature families: Lukasiewicz,
product, drastic, Schweizer-Sklar, Hamacher, Frank, Yager, Aczel-Alsina, Dombi,
and Sugeno-Weber.

`AdditiveGenerator` stores numerical grid checks in
`numerically_checked_properties`. Known generator families are marked
`guarantee="by construction"`.

## `balanced_fuzzy_sets.representable_uninorm_generator`

Builds an additive generator of a representable uninorm from a strict t-norm
additive generator and a neutral element `e`. It provides
`additive_generator_of_representable_uninorm(...)` and methods for evaluating
the generator, inverse, and uninorm.

## `balanced_fuzzy_sets.nullnorms`

Defines `Nullnorm` for operations `V:[0,1]^2->[0,1]` with zero element `z`.
It provides:

- `nullnorm_from_t_norm_and_t_conorm(...)`,
- `transport_nullnorm(...)`,
- `balanced_t_norm_from_nullnorm(...)`.

The general transport does not claim balanced symmetry. The balanced t-norm
construction adds the full balanced t-norm grid check after transport.

## `balanced_fuzzy_sets.rule_builder`

Parser for compact prefix-notation rules. Operator symbols:

- `N` - unary balanced fuzzy negation,
- `R` - unary supplementary operator,
- `T` - binary t-norm,
- `S` - binary t-conorm.

Unknown uppercase symbols are rejected as invalid operators. Extra arguments are
not ignored.

## `balanced_fuzzy_sets.transition_analysis`

Parser and evaluator for staged transition analysis. Stage file format:

```text
Stage1. x1=0.5, x2=0.7, x3=-0.2; 1: x1:=Rule1(x1); 2: x2:=Rule2(x2,x1,x3)
```

Rules have priorities and are executed from priority `1` upward. A rule result
updates the explicitly named output place on the left side of `:=`.

## `balanced_fuzzy_sets.petri_inspired_network`

Builds and draws a Petri-inspired transition graph. It is not a full formal
implementation of Petri-net semantics. Places are drawn as circles and
transitions as rectangles. The graph is layered:

```text
xk(1) -> rule transition -> xk(2) -> next rule transition -> xk(3)
```

There are no direct place-to-place edges. Every value update goes through a
transition corresponding to the fired rule.

## `balanced_fuzzy_sets.graphical_representation`

Plotting utilities for:

- negations and supplementary operators as 2D functions,
- t-norms, t-conorms, binary balanced operators, and binary rules as 3D
  surfaces,
- discontinuity-aware plotting that avoids connecting discontinuity points.

## Example Script

`examples/main_example.py` currently:

1. Selects the Aczel-Alsina additive generator with `p=3`.
2. Builds a representable uninorm with neutral element `e=0.5`.
3. Builds balanced t-norms and balanced t-conorms.
4. Builds a balanced negation and supplementary operator from the cosine fuzzy
   negation.
5. Loads `examples/data/rules/Rule1`, builds the rule, and plots it.
6. Saves a rule-value table for `x,y` from `-1` to `1` with step `0.1`.
7. Builds a three-step Petri-inspired transition graph from files in
   `examples/data/process_steps`.
8. Saves plots and summaries in `examples/generated`.

## Data Directories

`examples/data/rules` contains `Rule1` through `Rule10`, written in compact
prefix notation.

`examples/data/process_steps` contains `Step1` through `Step4`. Each file
describes initial place values and priority-ordered rule calls for one process
step.

`examples/data/transition_scenarios` contains independent `Scenario1` through
`Scenario4`. Each scenario has its own initial values and is not treated as a
continuation of the previous file.

`examples/generated` contains generated plots, the rule-value CSV table, and
the Petri-inspired transition graph summary.
