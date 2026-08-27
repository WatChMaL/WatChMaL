"""Shared pytest setup.

Kept deliberately thin: everything the checks need to *find* their subjects lives in
`tests/discovery.py` so that Tier 0 (which must run with pyyaml and nothing else) and
Tier 1 (which needs the framework installed) share one discovery layer.
"""

from tests.discovery import repo_root_on_syspath

repo_root_on_syspath()
