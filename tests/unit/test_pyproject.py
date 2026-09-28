"""Unit tests for the project manifest (pyproject.toml).

This file has no importable Python symbols: its "surface" is the TOML
document itself. These tests parse it directly and assert on the
declared project metadata, dependency set, and tool configuration that
every other spec's `verify` command (via `uv run ...`) relies on.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


def _find_pyproject() -> Path:
    """Locate the single project-root pyproject.toml.

    Walks upward from this test file looking for a pyproject.toml that
    declares the dental-clinic-platform project (rather than some
    unrelated ancestor manifest), so the test works regardless of the
    exact directory this file is ultimately collected from.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            try:
                data = tomllib.loads(candidate.read_text())
            except tomllib.TOMLDecodeError:
                continue
            if data.get("project", {}).get("name") == "dental-clinic-platform":
                return candidate
    raise FileNotFoundError(
        "Could not locate the dental-clinic-platform pyproject.toml "
        f"by walking up from {here}"
    )


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = _find_pyproject()
    return tomllib.loads(path.read_text())


_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _base_name(requirement: str) -> str:
    """Extract the bare package name from a PEP 508-ish requirement string.

    e.g. "sqlalchemy[asyncio]>=2.0" -> "sqlalchemy", "python-jose[cryptography]" -> "python-jose"
    """
    # Strip extras first (anything in [...]) then take the leading name token.
    without_extras = re.sub(r"\[[^\]]*\]", "", requirement)
    match = _NAME_RE.match(without_extras.strip())
    assert match, f"could not parse requirement: {requirement!r}"
    return match.group(1).lower()


def _requirement_for(deps: list[str], package: str) -> str:
    for dep in deps:
        if _base_name(dep) == package.lower():
            return dep
    raise AssertionError(f"dependency {package!r} not found in {deps!r}")


class TestProjectTable:
    def test_project_name(self, manifest):
        assert manifest["project"]["name"] == "dental-clinic-platform"

    def test_project_version(self, manifest):
        assert manifest["project"]["version"] == "1.0.0"

    def test_requires_python(self, manifest):
        assert manifest["project"]["requires-python"] == ">=3.12"


@pytest.fixture(scope="module")
def deps(manifest) -> list[str]:
    deps = manifest["project"]["dependencies"]
    assert isinstance(deps, list) and deps, "dependencies must be a non-empty list"
    return deps


@pytest.fixture(scope="module")
def dep_names(deps) -> set[str]:
    return {_base_name(d) for d in deps}


class TestRuntimeDependencies:

    @pytest.mark.parametrize(
        "package",
        [
            "fastapi",
            "uvicorn",
            "sqlalchemy",
            "asyncpg",
            "aiosqlite",
            "alembic",
            "pydantic",
            "pydantic-settings",
            "python-jose",
            "passlib",
            "aio-pika",
            "boto3",
            "python-multipart",
            "httpx",
        ],
    )
    def test_required_runtime_package_declared(self, dep_names, package):
        assert package in dep_names, f"missing required runtime dependency {package!r}"

    def test_uvicorn_declares_standard_extra(self, deps):
        req = _requirement_for(deps, "uvicorn")
        assert "[standard]" in req

    def test_sqlalchemy_declares_asyncio_extra_and_min_version(self, deps):
        req = _requirement_for(deps, "sqlalchemy")
        assert "[asyncio]" in req
        assert ">=2.0" in req

    def test_pydantic_min_version_2(self, deps):
        req = _requirement_for(deps, "pydantic")
        assert ">=2" in req

    def test_python_jose_declares_cryptography_extra(self, deps):
        req = _requirement_for(deps, "python-jose")
        assert "[cryptography]" in req

    def test_passlib_declares_bcrypt_extra(self, deps):
        req = _requirement_for(deps, "passlib")
        assert "[bcrypt]" in req


class TestDevDependencies:
    def _dev_list(self, manifest) -> list[str]:
        groups = manifest.get("dependency-groups", {})
        if "dev" in groups:
            return groups["dev"]
        optional = manifest.get("project", {}).get("optional-dependencies", {})
        if "dev" in optional:
            return optional["dev"]
        raise AssertionError(
            "no [dependency-groups].dev or [project.optional-dependencies].dev table found"
        )

    @pytest.mark.parametrize("package", ["pytest", "pytest-asyncio", "mypy", "ruff"])
    def test_dev_package_declared(self, manifest, package):
        dev_deps = self._dev_list(manifest)
        dev_names = {_base_name(d) for d in dev_deps}
        assert package in dev_names, f"missing dev dependency {package!r}"

    def test_no_redis_fake_dependency_pulled_in(self, manifest):
        # project_rules.testing's in-memory substitutes are hand-written in
        # app/core/*, not third-party packages: nothing redis/fakeredis-like
        # should appear anywhere in the declared dependency sets.
        dev_deps = self._dev_list(manifest)
        runtime_deps = manifest["project"]["dependencies"]
        all_names = {_base_name(d) for d in [*runtime_deps, *dev_deps]}
        assert not any("redis" in name for name in all_names)


class TestToolConfig:
    def test_ruff_target_version_py312(self, manifest):
        ruff_cfg = manifest["tool"]["ruff"]
        assert ruff_cfg["target-version"] == "py312"

    def test_ruff_line_length_is_set(self, manifest):
        ruff_cfg = manifest["tool"]["ruff"]
        assert isinstance(ruff_cfg.get("line-length"), int)
        assert ruff_cfg["line-length"] > 0

    def test_mypy_python_version(self, manifest):
        mypy_cfg = manifest["tool"]["mypy"]
        assert mypy_cfg["python_version"] == "3.12"

    def test_mypy_packages_includes_app(self, manifest):
        mypy_cfg = manifest["tool"]["mypy"]
        assert mypy_cfg["packages"] == ["app"]

    def test_pytest_asyncio_mode_auto(self, manifest):
        pytest_cfg = manifest["tool"]["pytest"]["ini_options"]
        assert pytest_cfg["asyncio_mode"] == "auto"

    def test_pytest_pythonpath_includes_project_root(self, manifest):
        pytest_cfg = manifest["tool"]["pytest"]["ini_options"]
        assert pytest_cfg["pythonpath"] == ["."]


class TestBuildSystem:
    def test_build_backend_is_hatchling_or_setuptools(self, manifest):
        build_system = manifest["build-system"]
        backend = build_system["build-backend"]
        assert backend in ("hatchling.build", "setuptools.build_meta"), (
            f"unexpected build backend {backend!r}, expected a minimal PEP 517 "
            "backend (hatchling or setuptools)"
        )

    def test_build_system_requires_matching_backend_package(self, manifest):
        build_system = manifest["build-system"]
        requires = build_system["requires"]
        backend = build_system["build-backend"]
        if backend == "hatchling.build":
            assert any(_base_name(r) == "hatchling" for r in requires)
        else:
            assert any(_base_name(r) == "setuptools" for r in requires)
