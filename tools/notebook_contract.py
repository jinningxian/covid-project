"""Data and execution contracts shared by the COVID-19 notebooks and tests."""

from __future__ import annotations

import math
from pathlib import Path


INPUT_COLUMNS = (
    "Sno",
    "Date",
    "Time",
    "State/UnionTerritory",
    "ConfirmedIndianNational",
    "ConfirmedForeignNational",
    "Cured",
    "Deaths",
    "Confirmed",
)
OUTPUT_SCHEMAS = {
    "clean_data.csv": set(INPUT_COLUMNS) | {"number of change"},
    "group_by_state.csv": {"State/UnionTerritory", "Cured", "Deaths", "Confirmed", "Cured Rate", "Deaths Rate", "Missing Rate"},
    "min_death_state_dataset.csv": set(INPUT_COLUMNS),
    "most_confirmed_state_dataset.csv": set(INPUT_COLUMNS),
    "most_cured_state_dataset.csv": set(INPUT_COLUMNS),
}


def validate_covid_frame(frame: object) -> None:
    """Fail closed before analysis when the source table cannot support it."""
    import numpy as np
    import pandas as pd

    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if len(frame) < 10:
        raise ValueError("at least 10 rows are required for a bounded train/test split")
    if frame["State/UnionTerritory"].nunique(dropna=True) < 2:
        raise ValueError("at least two states or territories are required")
    dates = pd.to_datetime(frame["Date"], format="%d/%m/%y", errors="coerce")
    if dates.isna().any():
        raise ValueError("Date must use DD/MM/YY and contain no invalid values")
    for column in ("Cured", "Deaths", "Confirmed"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numbers")
    for column in ("ConfirmedIndianNational", "ConfirmedForeignNational"):
        values = pd.to_numeric(frame[column].replace("-", "0"), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain numbers or '-' placeholders")


def validate_output_schemas(workspace: Path, expected_names: set[str]) -> dict[str, list[str]]:
    import pandas as pd

    if not expected_names <= set(OUTPUT_SCHEMAS):
        raise ValueError("unknown expected output")
    result: dict[str, list[str]] = {}
    for name in sorted(expected_names):
        path = workspace / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular output file: {name}")
        frame = pd.read_csv(path)
        columns = {column for column in frame.columns if not column.startswith("Unnamed:")}
        missing = OUTPUT_SCHEMAS[name] - columns
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        result[name] = list(frame.columns)
    return result


def finite_regression_metrics(metrics: dict[str, object]) -> bool:
    scalars = [metrics["rmse"], metrics["score"], metrics["intercept"], *metrics["coefficients"]]
    return all(math.isfinite(float(value)) for value in scalars)


def network_guard_source() -> str:
    """Return a sitecustomize guard inherited by the kernel and Python children."""
    return r'''
import ipaddress
import os
import socket

_covid_original_connect = socket.socket.connect
_covid_original_connect_ex = socket.socket.connect_ex
_covid_original_create_connection = socket.create_connection

def _covid_loopback(address):
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def _covid_deny(address):
    log = os.environ.get("COVID_NETWORK_DENIAL_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as stream:
            stream.write("DENIED\n")
    raise RuntimeError("external network is denied for notebook validation")

def _covid_connect(sock, address):
    if not _covid_loopback(address):
        return _covid_deny(address)
    return _covid_original_connect(sock, address)

def _covid_connect_ex(sock, address):
    if not _covid_loopback(address):
        return _covid_deny(address)
    return _covid_original_connect_ex(sock, address)

def _covid_create_connection(address, *args, **kwargs):
    if not _covid_loopback(address):
        return _covid_deny(address)
    return _covid_original_create_connection(address, *args, **kwargs)

socket.socket.connect = _covid_connect
socket.socket.connect_ex = _covid_connect_ex
socket.create_connection = _covid_create_connection
'''.lstrip()
