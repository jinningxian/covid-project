from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tomllib
import zipfile
from contextlib import contextmanager
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbclient import NotebookClient

from tools import bootstrap_locked_environment as bootstrap
from tools.notebook_contract import (
    INPUT_COLUMNS,
    finite_regression_metrics,
    network_guard_source,
    validate_covid_frame,
    validate_output_schemas,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    (ROOT / "Data Cleaning 1 - covid_19_india.csv.ipynb", 39, True),
    (ROOT / ".ipynb_checkpoints" / "Data Cleaning 1 - covid_19_india.csv-checkpoint.ipynb", 35, False),
)
TRACKED_CSV_HASHES = {
    "covid_19_india.csv": "0465b0a26c09585d35c471423aff78056f3f46bb527c149af7524511c1a9e4c9",
    "clean_data.csv": "f5f479e4962372f40884d8fd2d2fcabdbf81fc534553e86be7aeb06a96740184",
    "group_by_state.csv": "2a7c9191cb6dbf2c9fd8591195f1fc1fc8fc300324016549fb0f156f94e3d2ff",
    "min_death_state_dataset.csv": "1c305530bef9e4b7453a4164982d607380fef935e179f2b153407dc91f2d4993",
    "most_confirmed_state_dataset.csv": "f39b72e9aee0579b47e7b60acb1e2d1ca82ac332ac8f95c8642b294c538de418",
    "most_cured_state_dataset.csv": "c3de2e82ad1cac89693592e57eff86681803247d5a9a686ea4688b6c8b3ecd0f",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def environment(**updates: str):
    old = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def synthetic_frame() -> pd.DataFrame:
    return pd.read_csv(ROOT / "tests" / "fixtures" / "synthetic_covid.csv")


def test_all_74_actual_code_cells_parse() -> None:
    counts = []
    for path, expected, _ in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        actual = sum(cell.cell_type == "code" for cell in notebook.cells)
        assert actual == expected
        counts.append(actual)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:{cell.get('id', 'cell')}", "exec")
    assert sum(counts) == 74


def test_synthetic_fixture_has_exact_input_schema() -> None:
    frame = synthetic_frame()
    assert tuple(frame.columns) == INPUT_COLUMNS
    validate_covid_frame(frame)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda frame: frame.drop(columns=["Deaths"]), "missing required columns"),
        (lambda frame: frame.assign(Date="2021-not-a-date"), "Date must use"),
        (lambda frame: frame.assign(Confirmed=np.nan), "Confirmed must contain finite"),
        (lambda frame: frame.assign(Cured=np.inf), "Cured must contain finite"),
        (lambda frame: frame.iloc[:4], "at least 10 rows"),
    ],
)
def test_invalid_input_is_rejected(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_covid_frame(mutation(synthetic_frame()))


def test_read_only_input_remains_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "covid_19_india.csv"
    shutil.copyfile(ROOT / "tests" / "fixtures" / "synthetic_covid.csv", target)
    before = file_hash(target)
    target.chmod(stat.S_IREAD)
    frame = pd.read_csv(target)
    validate_covid_frame(frame)
    assert file_hash(target) == before


def test_network_guard_is_inherited_by_python_subprocess(tmp_path: Path) -> None:
    guard = tmp_path / "sitecustomize.py"
    guard.write_text(network_guard_source(), encoding="utf-8")
    log = tmp_path / "denials.log"
    code = "import socket\ntry: socket.create_connection(('example.com',443),.1)\nexcept RuntimeError: pass\nelse: raise SystemExit(2)"
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(tmp_path), "COVID_NETWORK_DENIAL_LOG": str(log), "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["DENIED"]


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_platform_lock_has_closed_56_package_graph(platform: str) -> None:
    lock = bootstrap.read_lock(ROOT / f"pylock.{platform}.toml")
    assert lock["platform"] == f"{platform}-x86_64"
    assert len(lock["packages"]) == 56
    assert all(item["wheel-url"].startswith("https://files.pythonhosted.org/") for item in lock["packages"])


def test_project_direct_dependencies_equal_lock_roots() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(project["project"]["dependencies"] + project["dependency-groups"]["dev"])
    expected = {
        "matplotlib==3.11.2", "numpy==2.5.3", "pandas==3.0.6", "plotly==7.1.0",
        "scikit-learn==1.9.1", "ipykernel==7.4.0", "nbclient==0.11.0",
        "nbformat==5.11.1", "pytest==9.1.1",
    }
    assert declared == expected


def test_tooling_manifest_is_pip_and_uv_free() -> None:
    manifest = bootstrap.read_tooling_manifest(ROOT)
    assert manifest["installer_mode"].startswith("exact hash-bound PyPA installer")
    assert manifest["sdists_allowed"] is False
    assert manifest["network_install"] is False
    assert [item["name"] for item in manifest["bootstrap"]] == ["installer", "packaging"]


def test_existing_wrong_hash_is_rejected_without_network(tmp_path: Path) -> None:
    wheel = tmp_path / "test.whl"
    wheel.write_bytes(b"bad")
    item = {
        "wheel-name": wheel.name,
        "wheel-url": "https://files.pythonhosted.org/packages/test.whl",
        "wheel-size": 3,
        "wheel-sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="existing wheel differs"):
        bootstrap.fetch_one(item, tmp_path)


@pytest.mark.parametrize("member", ["../escape.py", "D:../escaped-by-wheel.txt", "NUL.txt", "trailing./file.py"])
def test_unsafe_wheel_member_is_rejected(tmp_path: Path, member: str) -> None:
    wheel = tmp_path / (hashlib.sha256(member.encode()).hexdigest() + ".whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, "pass")
        archive.writestr("unsafe-1.dist-info/RECORD", f"{member},,\nunsafe-1.dist-info/RECORD,,\n")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        bootstrap.validate_record(wheel)


def test_opposite_platform_lock_is_rejected_before_install(tmp_path: Path) -> None:
    opposite = "linux" if sys.platform == "win32" else "windows"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools" / "bootstrap_locked_environment.py"),
         "--lock", str(ROOT / f"pylock.{opposite}.toml"), "--wheelhouse", str(tmp_path / "wheels"),
         "--target", str(tmp_path / "venv"), "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not (tmp_path / "venv").exists()


def test_notebooks_use_static_rendering_and_current_apis() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path, _, _ in NOTEBOOKS)
    assert "fig.show()" not in combined
    assert "write_image" not in combined
    assert "squared=True" not in combined
    assert combined.count("write_html") == 6
    assert combined.count("numeric_only=True") >= 8
    assert combined.count("root_mean_squared_error") == 4


def execute_notebook(path: Path, expected_count: int, includes_clean_output: bool, tmp_path: Path) -> dict[str, object]:
    short_root = os.environ.get("COVID_NOTEBOOK_EXECUTOR_ROOT")
    if short_root:
        workspace = Path(short_root) / ("main" if includes_clean_output else "checkpoint")
        if workspace.exists():
            raise ValueError(f"fresh notebook executor required: {workspace}")
        workspace.mkdir(parents=True)
    else:
        workspace = tmp_path / path.stem
        workspace.mkdir()
    input_path = workspace / "covid_19_india.csv"
    shutil.copyfile(ROOT / "tests" / "fixtures" / "synthetic_covid.csv", input_path)
    input_hash = file_hash(input_path)
    input_path.chmod(stat.S_IREAD)

    guard_dir = workspace / "network-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(network_guard_source(), encoding="utf-8")
    denial_log = workspace / "network-denials.log"
    notebook = nbformat.read(path, as_version=4)
    source_count = sum(cell.cell_type == "code" for cell in notebook.cells)
    assert source_count == expected_count
    validation_code = r'''
import json
import pathlib
import socket
import subprocess
import sys
import numpy as _np

assert len(X_train) + len(X_test) == len(X)
assert len(y_preds) == len(y_test) and len(y_test) > 0
_metrics = {
    "rows": int(len(X)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "prediction_rows": int(len(y_preds)),
    "rmse": float(RMSE),
    "score": float(linear_reg_model.score(X_test, y_test)),
    "coefficients": [float(value) for value in linear_reg.coef_],
    "intercept": float(linear_reg.intercept_),
}
assert _np.isfinite([_metrics["rmse"], _metrics["score"], _metrics["intercept"], *_metrics["coefficients"]]).all()
try:
    socket.create_connection(("example.com", 443), timeout=0.1)
except RuntimeError:
    pass
else:
    raise AssertionError("external network was not denied")
_child = "import socket\ntry: socket.create_connection(('example.com',443),.1)\nexcept RuntimeError: pass\nelse: raise SystemExit(2)"
subprocess.run([sys.executable, "-B", "-c", _child], check=True, timeout=10)
pathlib.Path("validation-metrics.json").write_text(json.dumps(_metrics, sort_keys=True), encoding="utf-8")
'''
    notebook.cells.append(nbformat.v4.new_code_cell(validation_code))
    python_path = os.pathsep.join([str(guard_dir), str(ROOT)])
    with environment(
        PYTHONPATH=python_path,
        PYTHONDONTWRITEBYTECODE="1",
        COVID_NETWORK_DENIAL_LOG=str(denial_log),
        MPLBACKEND="Agg",
    ):
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(workspace)}},
        )
        executed = client.execute()
    executed_source = sum(cell.cell_type == "code" and cell.get("execution_count") is not None for cell in executed.cells[:-1])
    assert executed_source == expected_count
    assert file_hash(input_path) == input_hash
    metrics = json.loads((workspace / "validation-metrics.json").read_text(encoding="utf-8"))
    assert metrics["rows"] == 36
    assert metrics["train_rows"] + metrics["test_rows"] == metrics["rows"]
    assert metrics["prediction_rows"] == metrics["test_rows"]
    assert finite_regression_metrics(metrics)
    expected_outputs = {
        "group_by_state.csv", "min_death_state_dataset.csv",
        "most_confirmed_state_dataset.csv", "most_cured_state_dataset.csv",
    }
    if includes_clean_output:
        expected_outputs.add("clean_data.csv")
    validate_output_schemas(workspace, expected_outputs)
    for name in ("correlation.html", "scatter_matrix.html", "state_correlation.html"):
        html = (workspace / name).read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert "<script src=" not in html.lower()
    assert (workspace / "regression.png").stat().st_size > 1_000
    assert denial_log.read_text(encoding="utf-8").splitlines() == ["DENIED", "DENIED"]
    return metrics


@pytest.mark.parametrize("path, expected_count, includes_clean_output", NOTEBOOKS)
def test_notebook_executes_all_actual_cells_offline(path: Path, expected_count: int, includes_clean_output: bool, tmp_path: Path) -> None:
    execute_notebook(path, expected_count, includes_clean_output, tmp_path)


def test_all_six_tracked_csvs_are_byte_identical_to_intake() -> None:
    assert {name: file_hash(ROOT / name) for name in TRACKED_CSV_HASHES} == TRACKED_CSV_HASHES
