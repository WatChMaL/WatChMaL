"""
Tier 0 - every config file is valid YAML.

Goal: no run can ever die on a stray tab or a bad indent. The repo ships ~110 config
files and is entirely config-driven, but until now nothing parsed them outside of an
actual launch - so a malformed config surfaced minutes into a cluster job, or after the
queue wait, rather than in the editor.

This is the cheapest possible gate and the only one that still works when the framework
itself cannot be installed, so it deliberately covers EVERY tree, including the ones
Tier 1 does not gate on yet.
"""

import pytest
import yaml

from tests.discovery import REPO_ROOT, all_config_files

CONFIG_FILES = all_config_files()


def test_config_files_are_discovered():
    """Guards the guard: an empty parametrisation would make this file pass vacuously."""
    assert CONFIG_FILES, "no YAML found under tutorial/ - has the config tree moved?"


@pytest.mark.parametrize(
    "config_file",
    CONFIG_FILES,
    ids=[str(p.relative_to(REPO_ROOT)) for p in CONFIG_FILES],
)
def test_yaml_parses(config_file):
    try:
        yaml.safe_load(config_file.read_text())
    except yaml.YAMLError as exc:
        pytest.fail(f"{config_file.relative_to(REPO_ROOT)} is not valid YAML:\n{exc}")
