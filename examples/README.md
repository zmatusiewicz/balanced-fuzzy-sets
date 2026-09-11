# Demonstration files

This directory is separate from the installable library in `src/`.

- `main.py` runs the complete demonstration.
- `data/rules/` contains example rules in compact prefix notation.
- `data/process_steps/` contains consecutive example process stages.
- `data/transition_scenarios/` contains independent example scenarios.
- `generated/` contains generated plots, CSV tables, and text summaries.

Run the demonstration from the repository root:

```bash
python examples/main.py
```

The script reads only from `data/` and writes its artifacts to `generated/`.
Files in `generated/` are reference outputs and may be overwritten when the
demonstration is run.
