Balanced Fuzzy Sets
===================

Balanced Fuzzy Sets is a Python toolkit for constructing, checking, composing,
evaluating, and visualizing fuzzy operators on the classical interval
``[0, 1]`` and the balanced interval ``[-1, 1]``.

Installation
------------

Install the package from the repository root:

.. code-block:: console

   python -m pip install -e .

The package requires Python 3.10 or newer. The tested versions are listed in
the project's ``pyproject.toml`` and GitHub Actions test matrix.

Quick start
-----------

Common classes and operations are exported directly from the package. Factory
families with overlapping names are available through their submodules.

.. code-block:: python

   from balanced_fuzzy_sets import BalancedFuzzyOperators
   from balanced_fuzzy_sets import tnorm_additive_generators

   generator = tnorm_additive_generators.aczel_alsina(3.0)
   balanced_t_norm = BalancedFuzzyOperators.balanced_t_norm_from_t_norm(
       generator.t_norm
   )
   print(balanced_t_norm(0.2, 0.3))

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
