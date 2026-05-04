from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def pytest_sessionfinish(session, exitstatus):
    results_dir = Path("allure-results")
    results_dir.mkdir(exist_ok=True)

    environment = {
        "Project": "Diabetes Prediction",
        "Python": sys.version.split()[0],
        "OS": platform.platform(),
        "pytest": _package_version("pytest"),
        "scikit-learn": _package_version("scikit-learn"),
        "pandas": _package_version("pandas"),
        "fastapi": _package_version("fastapi"),
        "allure-pytest": _package_version("allure-pytest"),
    }

    content = "\n".join(f"{key}={value}" for key, value in environment.items())

    (results_dir / "environment.properties").write_text(
        content,
        encoding="utf-8",
    )