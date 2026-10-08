# covid-project

Data cleaning and analysis of COVID-19 in India with two Jupyter notebooks.

## Reproducible environment

The project is pinned to CPython 3.13.16. Choose the lock for the host platform
and create a fresh environment without pip or uv:

```text
python -B tools/bootstrap_locked_environment.py \
  --lock pylock.windows.toml \
  --wheelhouse <task-owned-wheel-cache> \
  --target <fresh-environment> \
  --report <install-report.json> \
  --fetch
```

Use `pylock.linux.toml` on Linux. The fetch step accepts only the exact
`files.pythonhosted.org` URLs, sizes, and SHA-256 values in the lock. Wheel
installation then runs offline through PyPA `installer` 1.0.1, validates every
wheel `RECORD`, executes no source-distribution build scripts, and leaves pip,
uv, and installer out of the target environment.

## Validation

```text
<environment-python> -B -m pytest -q
```

Tests execute all 74 source notebook code cells against synthetic data in
copied workspaces. They deny external network use, use a non-interactive
matplotlib backend, emit self-contained Plotly HTML, check regression shapes
and finite metrics, and verify that the six tracked CSV files remain
byte-identical. The repository data files are never test output targets.
