"""Data and execution contracts shared by the COVID-19 notebooks and tests."""

from __future__ import annotations

import hashlib
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
ORIGINAL_CSV_BYTE_CONTRACT = {
    "covid_19_india.csv": {
        "git_lf": {
            "bytes": 472219,
            "sha256": "077a31be8f87dd5946f6fddefbebd04a2098158a12d31583d228552156c4d50a",
            "git_blob_oid": "75acefb2fc579d469b0fbb071f7cd42c65574e4b",
        },
        "windows_crlf": {
            "bytes": 481510,
            "sha256": "0465b0a26c09585d35c471423aff78056f3f46bb527c149af7524511c1a9e4c9",
        },
    },
    "clean_data.csv": {
        "git_lf": {
            "bytes": 560045,
            "sha256": "0461bee8968b314bc5fe13261ecee5623f12fef7208c5dffe6a99b2d42f29b71",
            "git_blob_oid": "384aeaaee849baa03c7b1307c60d6d549b46f31f",
        },
        "windows_crlf": {
            "bytes": 569337,
            "sha256": "f5f479e4962372f40884d8fd2d2fcabdbf81fc534553e86be7aeb06a96740184",
        },
    },
    "group_by_state.csv": {
        "git_lf": {
            "bytes": 7740,
            "sha256": "ece83c4afd9a31a4a9030939047e484f08cdb026c2044db7cec7ec6dbe1c4102",
            "git_blob_oid": "562a51b3fa640f25d9361cc60f3f9c029fe5d783",
        },
        "windows_crlf": {
            "bytes": 7786,
            "sha256": "2a7c9191cb6dbf2c9fd8591195f1fc1fc8fc300324016549fb0f156f94e3d2ff",
        },
    },
    "min_death_state_dataset.csv": {
        "git_lf": {
            "bytes": 4801,
            "sha256": "31c86637db444479e9cf1cbae3df3572cfb45cfd06789cd31d013a2be6fb8e5c",
            "git_blob_oid": "bdabbb9b9425070513062379a56ceb4becd36502",
        },
        "windows_crlf": {
            "bytes": 4862,
            "sha256": "1c305530bef9e4b7453a4164982d607380fef935e179f2b153407dc91f2d4993",
        },
    },
    "most_confirmed_state_dataset.csv": {
        "git_lf": {
            "bytes": 199,
            "sha256": "f6cfa5b8fa98c8e54cf5715c52f2b38b73532e26e648159436a3e25e9224f518",
            "git_blob_oid": "0d8b2c060adf2de66fa2c54eb2ce54c1ec22a825",
        },
        "windows_crlf": {
            "bytes": 201,
            "sha256": "f39b72e9aee0579b47e7b60acb1e2d1ca82ac332ac8f95c8642b294c538de418",
        },
    },
    "most_cured_state_dataset.csv": {
        "git_lf": {
            "bytes": 190,
            "sha256": "22ba140825655b1ee361bce863afa6dd75d5d9c29c115aee89b052c36f26b1e5",
            "git_blob_oid": "0dfb89706f8ba109f889958b557fcae8f67f9ba4",
        },
        "windows_crlf": {
            "bytes": 192,
            "sha256": "c3de2e82ad1cac89693592e57eff86681803247d5a9a686ea4688b6c8b3ecd0f",
        },
    },
}


def identify_original_csv_representation(name: str, payload: bytes) -> str:
    """Return the exact frozen representation without normalizing payload bytes."""
    contract = ORIGINAL_CSV_BYTE_CONTRACT.get(name)
    if contract is None:
        raise ValueError(f"unknown original CSV: {name}")
    digest = hashlib.sha256(payload).hexdigest()
    matches = [
        representation
        for representation, expected in contract.items()
        if len(payload) == expected["bytes"] and digest == expected["sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{name} does not match either exact frozen representation")
    representation = matches[0]
    if representation == "git_lf":
        git_hash = hashlib.sha1(usedforsecurity=False)
        git_hash.update(f"blob {len(payload)}\0".encode("ascii"))
        git_hash.update(payload)
        if git_hash.hexdigest() != contract["git_lf"]["git_blob_oid"]:
            raise ValueError(f"{name} does not match its frozen Git blob OID")
    return representation


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
