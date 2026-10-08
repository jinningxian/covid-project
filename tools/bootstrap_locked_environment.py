"""Create a fresh, pip-free environment from a hash-bound platform lock.

The fetch step is deliberately separate from the offline installation step.
Only HTTPS files.pythonhosted.org URLs declared in the checked-in lock and
tooling manifest are accepted.  Installation validates the outer SHA-256 and
every wheel RECORD entry before invoking PyPA installer with network denied.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import runpy
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_PYTHON = (3, 13, 16)
EXPECTED_DIRECT = {
    "ipykernel",
    "matplotlib",
    "nbclient",
    "nbformat",
    "numpy",
    "pandas",
    "plotly",
    "pytest",
    "scikit-learn",
}
LOCK_KEYS = {"lock-version", "created-by", "requires-python", "platform", "packages"}
PACKAGE_KEYS = {
    "name",
    "version",
    "direct",
    "wheel-name",
    "wheel-url",
    "wheel-size",
    "wheel-sha256",
    "primary-metadata-url",
    "primary-metadata-sha256",
}
WINDOWS_DEVICE_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{index}" for index in range(1, 10)},
    *{f"LPT{index}" for index in range(1, 10)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def read_lock(path: Path) -> dict[str, object]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(data) != LOCK_KEYS:
        raise ValueError(f"unexpected lock keys: {sorted(set(data) ^ LOCK_KEYS)}")
    if data["lock-version"] != "1.0" or data["requires-python"] != "==3.13.16":
        raise ValueError("unsupported lock or Python version")
    packages = data["packages"]
    if not isinstance(packages, list) or len(packages) != 56:
        raise ValueError("the platform lock must contain exactly 56 packages")
    names: set[str] = set()
    direct: set[str] = set()
    for item in packages:
        if not isinstance(item, dict) or set(item) != PACKAGE_KEYS:
            raise ValueError("package entry does not match the closed lock schema")
        name = canonical_name(str(item["name"]))
        if name in names:
            raise ValueError(f"duplicate locked package: {name}")
        names.add(name)
        if item["direct"] is True:
            direct.add(name)
        url = urllib.parse.urlparse(str(item["wheel-url"]))
        if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
            raise ValueError(f"untrusted wheel URL for {name}")
        if Path(url.path).name != item["wheel-name"]:
            raise ValueError(f"wheel URL/name mismatch for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["wheel-sha256"])):
            raise ValueError(f"invalid wheel hash for {name}")
        if not isinstance(item["wheel-size"], int) or item["wheel-size"] <= 0:
            raise ValueError(f"invalid wheel size for {name}")
    if direct != EXPECTED_DIRECT:
        raise ValueError(f"direct dependency set changed: {sorted(direct ^ EXPECTED_DIRECT)}")
    return data


def read_tooling_manifest(repo_root: Path) -> dict[str, object]:
    path = repo_root / "tools" / "locked_tooling_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {
        "bootstrap",
        "installer_mode",
        "locks",
        "network_install",
        "python",
        "schema",
        "sdists_allowed",
        "scripts_enabled",
    }:
        raise ValueError("tooling manifest has unexpected keys")
    if data["schema"] != 1 or data["python"] != "3.13.16":
        raise ValueError("unsupported tooling manifest")
    if data["sdists_allowed"] is not False or data["network_install"] is not False:
        raise ValueError("sdists and network installation must stay disabled")
    bootstrap = data["bootstrap"]
    if [item["name"] for item in bootstrap] != ["installer", "packaging"]:
        raise ValueError("bootstrap set must be installer then packaging")
    return data


def validate_record(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ZIP member in {path.name}")
        for name in names:
            pure = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or name.startswith("/")
                or "\x00" in name
                or any(
                    part in {"", ".", ".."}
                    or ":" in part
                    or part.endswith((" ", "."))
                    or part.split(".", 1)[0].upper() in WINDOWS_DEVICE_NAMES
                    for part in pure.parts
                )
            ):
                raise ValueError(f"unsafe ZIP member in {path.name}")
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError(f"expected one RECORD in {path.name}")
        rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8"))))
        mapping: dict[str, tuple[str, str]] = {}
        for row in rows:
            if len(row) != 3 or row[0] in mapping:
                raise ValueError(f"invalid or duplicate RECORD row in {path.name}")
            mapping[row[0]] = (row[1], row[2])
        if set(names) != set(mapping):
            raise ValueError(f"RECORD membership mismatch in {path.name}")
        for name in names:
            hash_field, size_field = mapping[name]
            if name == records[0]:
                if hash_field or size_field:
                    raise ValueError(f"RECORD self-row must be unsigned in {path.name}")
                continue
            if not hash_field or not size_field:
                raise ValueError(f"missing RECORD hash or size in {path.name}")
            algorithm, encoded = hash_field.split("=", 1)
            if algorithm not in {"sha256", "sha384", "sha512"}:
                raise ValueError(f"weak RECORD hash in {path.name}")
            payload = archive.read(name)
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, payload).digest()).rstrip(b"=").decode("ascii")
            if actual != encoded or len(payload) != int(size_field):
                raise ValueError(f"RECORD mismatch in {path.name}")


def ensure_regular_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"wheelhouse must be a real directory: {path}")


def fetch_one(item: dict[str, object], wheelhouse: Path) -> Path:
    name = str(item.get("wheel-name", item.get("filename")))
    url = str(item["wheel-url"] if "wheel-url" in item else item["url"])
    expected_size = int(item["wheel-size"] if "wheel-size" in item else item["size"])
    expected_hash = str(item["wheel-sha256"] if "wheel-sha256" in item else item["sha256"])
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org" or Path(parsed.path).name != name:
        raise ValueError(f"untrusted download URL: {url}")
    destination = wheelhouse / name
    if destination.is_symlink():
        raise ValueError(f"wheel path must not be a symlink: {destination}")
    if destination.exists():
        if destination.stat().st_size != expected_size or sha256(destination) != expected_hash:
            raise ValueError(f"existing wheel differs from lock: {name}")
        validate_record(destination)
        return destination
    fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=wheelhouse)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size != expected_size or sha256(temporary) != expected_hash:
            raise ValueError(f"downloaded wheel differs from lock: {name}")
        validate_record(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def locate_wheels(lock: dict[str, object], tooling: dict[str, object], wheelhouse: Path) -> tuple[list[Path], dict[str, Path]]:
    targets: list[Path] = []
    tools: dict[str, Path] = {}
    for item in tooling["bootstrap"]:
        path = wheelhouse / item["filename"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"missing or changed bootstrap wheel: {item['name']}")
        validate_record(path)
        tools[item["name"]] = path
    for item in lock["packages"]:
        path = wheelhouse / item["wheel-name"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["wheel-size"] or sha256(path) != item["wheel-sha256"]:
            raise ValueError(f"missing or changed locked wheel: {item['name']}")
        validate_record(path)
        targets.append(path)
    return targets, tools


def install_current(lock: dict[str, object], tooling: dict[str, object], wheelhouse: Path, report: Path) -> None:
    if sys.prefix == sys.base_prefix:
        raise ValueError("offline installation requires a fresh virtual environment")
    before = list(importlib.metadata.distributions())
    if before:
        raise ValueError("target environment is not empty")
    targets, tools = locate_wheels(lock, tooling, wheelhouse)

    sys.path.insert(0, str(tools["packaging"]))
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename

    accepted_tags = set(sys_tags())
    for target in targets:
        _, _, _, wheel_tags = parse_wheel_filename(target.name)
        if not accepted_tags.intersection(wheel_tags):
            raise ValueError(f"incompatible wheel for this interpreter: {target.name}")

    socket_events: list[str] = []

    def deny_network(event: str, _args: tuple[object, ...]) -> None:
        if event.startswith("socket."):
            socket_events.append(event)
            raise RuntimeError("network is forbidden during wheel installation")

    sys.addaudithook(deny_network)
    sys.path.insert(0, str(tools["installer"]))
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m installer",
            "--validate-record",
            "all",
            "--no-compile-bytecode",
            *[str(path) for path in targets],
        ]
        runpy.run_module("installer", run_name="__main__")
    finally:
        sys.argv = original_argv

    # The bootstrap wheels are zip-imported tools, not installed target
    # distributions.  Remove their temporary sys.path entries before
    # enumerating the physical environment so they cannot inflate the graph.
    bootstrap_paths = {str(path) for path in tools.values()}
    sys.path[:] = [entry for entry in sys.path if entry not in bootstrap_paths]
    distributions = list(importlib.metadata.distributions())
    actual = {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in distributions
        if dist.metadata.get("Name")
    }
    expected = {canonicalize_name(item["name"]): item["version"] for item in lock["packages"]}
    if actual != expected:
        raise ValueError(f"installed graph differs from lock: expected={len(expected)} actual={len(actual)}")
    if {"pip", "uv", "installer"}.intersection(actual):
        raise ValueError("forbidden installer/resolver distribution present in target")

    marker_environment = default_environment()
    marker_environment["extra"] = ""
    active_edges: list[dict[str, str]] = []
    for dist in distributions:
        parent = canonicalize_name(dist.metadata["Name"])
        for raw_requirement in dist.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker and not requirement.marker.evaluate(marker_environment):
                continue
            child = canonicalize_name(requirement.name)
            if child not in actual or actual[child] not in requirement.specifier:
                raise ValueError(f"unsatisfied locked dependency: {parent} -> {raw_requirement}")
            active_edges.append({"from": parent, "to": child, "requirement": raw_requirement})

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "result": "PASS",
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "lock_platform": lock["platform"],
                "package_count": len(actual),
                "active_edge_count": len(active_edges),
                "packages": sorted(actual.items()),
                "active_edges": sorted(active_edges, key=lambda item: (item["from"], item["to"], item["requirement"])),
                "pip_uv_installer_installed": False,
                "wheel_record_validation": "all",
                "external_request_attempts_during_install": len(socket_events),
                "scripts_or_sdists_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--install-current", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise ValueError(f"exact CPython 3.13.16 required; found {sys.version.split()[0]}")
    repo_root = Path(__file__).resolve().parents[1]
    lock = read_lock(args.lock.resolve())
    tooling = read_tooling_manifest(repo_root)
    expected_platform = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64" if sys.platform.startswith("linux") else None
    if lock["platform"] != expected_platform:
        raise ValueError(f"lock platform {lock['platform']} does not match {sys.platform}")
    # Preserve an explicitly supplied short drive or mount path.  Resolving a
    # Windows SUBST path expands it back to the long backing path and can make
    # otherwise valid wheel members exceed MAX_PATH during installation.
    wheelhouse = args.wheelhouse.absolute()
    ensure_regular_directory(wheelhouse)
    if args.install_current:
        install_current(lock, tooling, wheelhouse, args.report.absolute())
        return
    if args.target is None:
        raise ValueError("--target is required when creating an environment")
    if args.fetch:
        for item in tooling["bootstrap"]:
            fetch_one(item, wheelhouse)
        for item in lock["packages"]:
            fetch_one(item, wheelhouse)
    locate_wheels(lock, tooling, wheelhouse)
    target = args.target.absolute()
    if target.exists():
        raise ValueError(f"fresh target required: {target}")
    subprocess.run([sys.executable, "-B", "-m", "venv", "--without-pip", str(target)], check=True)
    target_python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            str(target_python),
            "-B",
            str(Path(__file__).resolve()),
            "--lock",
            str(args.lock.resolve()),
            "--wheelhouse",
            str(wheelhouse),
            "--report",
            str(args.report.absolute()),
            "--install-current",
        ],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
