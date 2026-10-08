# covid-project 0.1.0 security handover

The historical migration base is commit
`a7ab6a9dc80f9a230dbc9e10504de4dfb865513b`. Commit, review, merge, and release
state are intentionally kept in the external per-commit evidence ledger so
this document remains accurate after normal Git transitions.

## Delivered behavior

- CPython is fixed at 3.13.16.
- Runtime dependencies are fixed at matplotlib 3.11.2, NumPy 2.5.3,
  pandas 3.0.6, Plotly 7.1.0, and scikit-learn 1.9.1.
- Notebook/test dependencies are fixed at ipykernel 7.4.0, nbclient 0.11.0,
  nbformat 5.11.1, and pytest 9.1.1.
- Windows and Linux each have a 56-package, wheel-only lock containing the
  exact PyPI URL, byte size, and SHA-256 for every selected artifact.
- A fresh environment is created without pip or uv. PyPA installer 1.0.1 and
  packaging 26.3 are hash-bound bootstrap tools; every wheel `RECORD` is
  validated and the installed dependency/marker closure must equal the lock.
- Both notebooks retain their 39 and 35 source code cells. They use pandas 3
  numeric aggregation, the current scikit-learn RMSE API, static Plotly HTML,
  and a non-interactive matplotlib PNG.
- Validation uses a synthetic 36-row, three-state table. It never writes the
  six tracked CSV files.

## Validation contract

The required local matrix is:

1. Create a fresh Windows environment from `pylock.windows.toml` and a fresh
   Linux environment from `pylock.linux.toml`.
2. Confirm exactly 56 installed distributions, all active dependency edges,
   no pip/uv/installer distribution, all wheel outer hashes and all `RECORD`
   entries.
3. Run `python -B -m pytest -q -p no:cacheprovider`; the suite executes all 74
   source cells, validates six CSV schemas, finite regression metrics and
   static output, and tests missing columns, invalid dates, NaN/Inf,
   too-small samples, read-only input, bad wheel hashes, unsafe members
   (including Windows drive-relative/device/trailing-dot paths), incompatible
   locks, and denied network.
4. Run Linux in a container with `--network none`. The Windows test guard also
   denies external sockets in the kernel and inherited Python subprocesses
   while permitting loopback ZMQ.
5. Recheck all six tracked CSV byte hashes against the intake.

The final functional evidence recorded 23/23 tests on Windows and 23/23 on Linux.
The Windows environment had 78 active edges and the Linux environment 77.
The Linux source mount was read-only. A Windows ZMQ proactor warning is an
environment diagnostic; it did not skip a test or weaken a threshold.

On Windows, use a short task-owned target path. Two preserved early attempts
show that expanding the target back to the long evidence path can exceed
Windows path limits for deep members in upstream Jedi/debugpy wheels.

## Security coverage and residuals

The exact runtime, development, bootstrap, and requirement/tag-validation set
contains 59 unique PyPI name/version coordinates. Batch OSV queries and PyPI
release metadata returned zero affected records for those exact coordinates.
No advisory was suppressed or dismissed.

This scoped result is not whole-physical-graph zero. Thirty selected wheels
contain native members, and PyPI metadata does not completely identify all
bundled native component versions. CPython 3.13.16 and the Windows/Linux base
runtime are also outside PyPI package-coordinate coverage. Those items remain
an explicit follow-up for component/SBOM and platform advisory mapping.

The three CPython 3.12 `.pyc` files accidentally created during a local syntax
check are retained byte-for-byte and ignored by the two narrow Python cache
rules. They are not source, are not staged, and cannot be loaded by the
CPython 3.13 validation commands. Their paths and hashes are recorded in the
external incident receipt.

## Rollback

Revert the declared dependency, notebook, test, tooling, README, version, and
handover paths as one batch. The tracked CSV, Tableau, and presentation assets
do not require data rollback because this migration does not modify them.
