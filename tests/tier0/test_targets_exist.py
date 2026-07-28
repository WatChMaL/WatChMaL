"""
Tier 0 - every in-repo `_target_` names something that exists.

Goal: catch the class of breakage that a config-driven framework is most exposed to and
that no compiler can see - a config pointing at a class or function that was renamed,
moved or never existed. `python -m compileall` is happy, the import graph is happy, and
the run dies at instantiation time.

This is the *static* half of that guard: it matches each `_target_` against the module
files by AST, importing nothing, so it runs with no framework dependency installed and
names the offending config and module instead of raising an ImportError. Tier 1 proves
the same property properly (by actually resolving the target through hydra) for the
trees it gates on; this one covers every tree in the repo and still works when the
dependencies do not install at all.

Re-exports count: `from watchmal.utils.multiring_sparse_helpers import build_model_...`
makes that name a legitimate `_target_` suffix on the importing module, and the
multi-ring test config relies on exactly that.

Targets pointing outside the repo (`torch.optim.Adam`, ...) are not checked here - they
are third-party API, verified by Tier 1's real resolution in an environment that has
them installed.
"""

import pytest
import yaml

from tests.discovery import (
    REPO_ROOT,
    all_config_files,
    is_in_repo_target,
    iter_targets,
    resolve_target_as_text,
)

LEDGER_PATH = REPO_ROOT / "tests" / "data" / "known_broken_targets.txt"


def _load_ledger() -> set[str]:
    if not LEDGER_PATH.is_file():
        return set()
    return {
        line.strip()
        for line in LEDGER_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


KNOWN_BROKEN = _load_ledger()


def _collect():
    """(config, target) for every in-repo `_target_`, deduplicated per config."""
    pairs = []
    for config_file in all_config_files():
        try:
            document = yaml.safe_load(config_file.read_text())
        except yaml.YAMLError:
            continue  # reported by test_yaml_parses.py; not this check's business
        seen = set()
        for target, _key_path in iter_targets(document):
            if is_in_repo_target(target) and target not in seen:
                seen.add(target)
                pairs.append((config_file, target))
    return pairs


def _param(config_file, target):
    rel = config_file.relative_to(REPO_ROOT)
    marks = ()
    if target in KNOWN_BROKEN:
        marks = pytest.mark.xfail(
            strict=True,
            reason=(
                f"{target} is in tests/data/known_broken_targets.txt - a pre-existing "
                "stale config reference. Fixing it must also delete its line there."
            ),
        )
    return pytest.param(config_file, target, marks=marks, id=f"{rel}::{target}")


TARGET_PARAMS = [_param(config_file, target) for config_file, target in _collect()]


def test_targets_are_discovered():
    assert TARGET_PARAMS, "no in-repo `_target_` found - has the config tree moved?"


@pytest.mark.parametrize("config_file, target", TARGET_PARAMS)
def test_target_resolves_statically(config_file, target):
    reason = resolve_target_as_text(target)
    assert reason is None, (
        f"{config_file.relative_to(REPO_ROOT)} points at `{target}` but {reason}"
    )


def test_ledger_has_no_stale_entries():
    """A ledger entry whose config is gone must be deleted, or the list quietly becomes
    a lie that nobody can act on."""
    referenced = {target for _config, target in _collect()}
    stale = sorted(KNOWN_BROKEN - referenced)
    assert not stale, (
        "these entries in tests/data/known_broken_targets.txt are no longer referenced "
        f"by any config and must be removed: {stale}"
    )
