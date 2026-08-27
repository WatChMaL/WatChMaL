"""
Backward-compatibility shim.

The unified core has a single logging module, ``watchmal.utils.logging_utils``. This
file used to carry a duplicate copy for the caverns core; it now simply re-exports the
canonical names so existing `from watchmal.utils.logging_utils_caverns import ...`
imports keep working. Prefer importing from ``watchmal.utils.logging_utils`` in new code.
"""

from watchmal.utils.logging_utils import (  # noqa: F401
    CSVLog,
    DisplayFilter,
    get_git_version,
    setup_logging,
    setup_logging_with_filter,
)

__all__ = [
    "CSVLog",
    "DisplayFilter",
    "get_git_version",
    "setup_logging",
    "setup_logging_with_filter",
]
