"""
Tier 1 - every entry config composes, resolves and carries the keys the worker reads.

Goal: this framework's complexity lives in its config composition - `@package`
redirects, group defaults across a tree, `hydra.searchpath` pointing at the tree root -
and none of it is exercised by any import. A deleted group file, a renamed class behind
a `_target_` or a missing key produces a perfectly valid Python program that dies on the
cluster after the queue wait. Composition needs no data, no GPU and instantiates
nothing, so proving it is nearly free.

Three properties, in increasing strength:

* **composes** - the defaults list resolves and the tree is self-consistent;
* **resolves** - every `_target_` in the composed result names a class or function that
  can actually be imported (this is the real guard for renames: Tier 0's static scan can
  only see text, this one imports);
* **conforms to the entrypoint** - the top-level keys `run.py` and `main.py` read by
  attribute are present, since an attribute read on a hydra config raises rather than
  returning None.

Scope: the config trees Tier 1 gates on - see `TIER1_EXCLUDED_CONFIG_TREES` in
tests/discovery.py. Both the trees and their entry configs are discovered by shape, so
when the two trees merge into one this file needs no edit.
"""

import pytest
import yaml

from tests.discovery import (
    REPO_ROOT,
    TIER1_EXCLUDED_CONFIG_TREES,
    compose_entry,
    config_trees,
    entry_configs,
    in_scope_config_trees,
    iter_targets,
    missing_optional_module,
    required_top_level_keys,
)

# Retirement trigger for the @pytest.mark.transitional checks at the bottom of the file.
RETIRES_WITH = "P5 - one config tree, and the `core:` key removed from configs"

ENTRY_CONFIGS = entry_configs()
ENTRY_IDS = [str(p.relative_to(REPO_ROOT)) for p in ENTRY_CONFIGS]


def test_entry_configs_are_discovered():
    assert ENTRY_CONFIGS, (
        "no entry config found in the in-scope trees "
        f"({[t.name for t in in_scope_config_trees()]}) - has the config tree moved?"
    )


@pytest.mark.parametrize("entry", ENTRY_CONFIGS, ids=ENTRY_IDS)
def test_config_composes(entry):
    """Compose exactly as the launch scripts do."""
    config = compose_entry(entry)
    assert config is not None


@pytest.mark.parametrize("entry", ENTRY_CONFIGS, ids=ENTRY_IDS)
def test_targets_resolve(entry):
    """Import every `_target_` the composed config would instantiate.

    Targets whose module needs an optional dependency that is not installed are skipped:
    that is the graph job's business, not the core job's.
    """
    from hydra.errors import InstantiationException
    from omegaconf import OmegaConf

    try:
        get_object = __import__("hydra.utils", fromlist=["get_object"]).get_object
    except (ImportError, AttributeError):  # hydra < 1.3
        from hydra.utils import get_class as get_object

    config = OmegaConf.to_container(compose_entry(entry), resolve=False)

    failures = []
    skipped = []
    for target, key_path in iter_targets(config):
        try:
            get_object(target)
        except (ImportError, InstantiationException, ValueError) as exc:
            missing = missing_optional_module(exc)
            if missing:
                skipped.append(f"{target} ({missing} not installed)")
                continue
            failures.append(f"{key_path}: {target} -> {type(exc).__name__}: {exc}")

    assert not failures, (
        f"{entry.relative_to(REPO_ROOT)} points at targets that do not resolve:\n  "
        + "\n  ".join(failures)
    )
    if skipped and not failures:
        # Nothing was proven for this config in this environment; say so rather than
        # reporting a green that means "not checked here".
        pytest.skip("optional dependency missing: " + "; ".join(skipped))


@pytest.mark.parametrize("entry", ENTRY_CONFIGS, ids=ENTRY_IDS)
def test_config_has_the_keys_the_entrypoints_read(entry):
    """Required keys are derived from run.py / main.py, never listed here."""
    required = required_top_level_keys()
    config = compose_entry(entry)
    missing = {key: where for key, where in required.items() if key not in config}
    assert not missing, (
        f"{entry.relative_to(REPO_ROOT)} is missing top-level key(s) "
        f"{sorted(missing)}, read unguarded by the entrypoint at "
        f"{sorted(missing.values())}. Either the config gains the key, or the "
        "entrypoint should read it through .get() with a default."
    )


@pytest.mark.parametrize("entry", ENTRY_CONFIGS, ids=ENTRY_IDS)
def test_master_port_is_usable_when_set(entry):
    """MASTER_PORT is optional (run.py defaults it), but if a config sets it, the value
    - offset by the first GPU index, as the worker does - must be a legal port."""
    config = compose_entry(entry)
    if "MASTER_PORT" not in config:
        return
    gpu_list = config.get("gpu_list") or []
    offset = int(gpu_list[0]) if gpu_list else 0
    port = int(config.MASTER_PORT) + offset
    assert 1024 < port < 65536, (
        f"{entry.relative_to(REPO_ROOT)}: MASTER_PORT {config.MASTER_PORT} + gpu_list[0] "
        f"offset {offset} = {port}, which is not a usable rendez-vous port."
    )


# --------------------------------------------------------------------------------- #
# Transitional - these exist to die. `pytest -m transitional` lists what is left.
# --------------------------------------------------------------------------------- #

def _configs_declaring_core():
    """Entry configs still carrying the obsolete `core:` key.

    Parametrising over this makes the check self-retiring: when the last `core:` key is
    stripped the parametrisation is empty, the test disappears and nothing has to be
    remembered.
    """
    found = []
    for entry in ENTRY_CONFIGS:
        try:
            document = yaml.safe_load(entry.read_text()) or {}
        except yaml.YAMLError:
            continue
        if "core" in document:
            found.append(entry)
    return found


CORE_KEY_CONFIGS = _configs_declaring_core()


@pytest.mark.transitional
@pytest.mark.parametrize(
    "entry",
    CORE_KEY_CONFIGS,
    ids=[str(p.relative_to(REPO_ROOT)) for p in CORE_KEY_CONFIGS],
)
def test_core_key_is_tolerated_but_never_read(entry):
    """`core:` selected the entrypoint before the merge; it is now dead config.

    Until it is stripped from the configs it must stay harmless - composing must not
    break, and no entrypoint may read it again. A config key that is silently ignored is
    fine; a config key that comes back to life is a return of the two-core split.
    """
    config = compose_entry(entry)
    assert "core" in config
    assert "core" not in required_top_level_keys(), (
        "an entrypoint reads `core` again - the single-core merge is being undone"
    )
    for path in (REPO_ROOT / "main.py", REPO_ROOT / "watchmal" / "entrypoints" / "run.py"):
        source = path.read_text()
        assert '"core"' not in source and "'core'" not in source, (
            f"{path.name} references the `core` config key again"
        )


@pytest.mark.transitional
def test_excluded_config_trees_still_exist():
    """The Tier 1 scope exclusion is the suite's only hardcoded config name.

    When the excluded tree disappears - i.e. when the trees merge - this fails, which is
    the reminder to delete `TIER1_EXCLUDED_CONFIG_TREES` and get the whole tree gated
    for free. Without it the exclusion would quietly outlive its reason.
    """
    present = {tree.name for tree in config_trees()}
    vanished = sorted(TIER1_EXCLUDED_CONFIG_TREES - present)
    assert not vanished, (
        f"config tree(s) {vanished} no longer exist, so excluding them from Tier 1 is "
        "meaningless: delete them from TIER1_EXCLUDED_CONFIG_TREES in "
        "tests/discovery.py (and this test with the last one)."
    )
