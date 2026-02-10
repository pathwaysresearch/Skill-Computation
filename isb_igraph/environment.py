from __future__ import annotations

import os
import sys


def assert_runtime_compatibility() -> None:
    if os.getenv("ISB_IGRAPH_SKIP_RUNTIME_CHECK", "0") == "1":
        return

    if sys.version_info < (3, 11):
        raise RuntimeError(
            "Unsupported Python version. Use Python 3.11+ for this project "
            f"(current: {sys.version.split()[0]})."
        )

    try:
        import igraph  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "python-igraph is not installed or broken in this environment. "
            "Install dependencies via `pip install -r requirements.txt`."
        ) from exc
