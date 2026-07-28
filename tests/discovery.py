"""
Shared discovery helpers for the CI checks.

Every check in `tests/` is built on one rule, which is what makes the suite survive the
ongoing core unification:

    A check asserts a property of the target architecture, DISCOVERS its subjects at
    runtime, and routes through the entrypoint-facing API rather than family-specific
    internals.

Concretely that means no check hardcodes an inventory. Modules are found by walking the
package, configs by globbing the config tree, config trees themselves by shape rather
than by name, and the worker -> engine contract by reading `watchmal/entrypoints/run.py`
with `ast` instead of keeping a copy of its call list. The upcoming unification steps
rename classes, merge the two config trees and retire shim modules - all of those are
*names and locations*, never *behaviours*, so a check that discovers names and asserts
behaviour cannot be broken by them. Better: a discovery-based check automatically covers
engines, configs and modules that do not exist yet.

The one deliberate exception is `TIER1_EXCLUDED_CONFIG_TREES` - see there.
"""

from __future__ import annotations

import ast
import importlib
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages holding the code under test. Discovered rather than listed inside, but the
# roots themselves have to start somewhere.
SOURCE_PACKAGES = ("watchmal", "analysis")

CONFIG_ROOT = REPO_ROOT / "tutorial" / "config"

# The single entrypoint whose contract with the engines is the definition of "one core".
RUN_PY = REPO_ROOT / "watchmal" / "entrypoints" / "run.py"

# --------------------------------------------------------------------------------- #
# Transitional scope
# --------------------------------------------------------------------------------- #

# Config trees that Tier 1 (hydra compose / target resolution / schema) does NOT check
# yet, by explicit decision: the ex-watchmal tree is inherited from upstream and is
# scheduled to be reworked or merged away, so gating PRs on it today would report debt
# that nobody is fixing this week. Tier 0 still parses every YAML in it.
#
# This is the suite's only hardcoded config name. It is written to be self-deleting:
# `test_config_compose.py::test_excluded_config_trees_still_exist` fails loudly the day
# the named tree disappears (i.e. when the trees merge), which is the signal to delete
# this constant and get the whole tree checked for free.
#
# RETIRES_WITH: P5 - one config tree.
TIER1_EXCLUDED_CONFIG_TREES = frozenset({"watchmal"})

# Optional dependencies that are documented in an optional requirements file only as a
# comment, because the distribution name is not the import name or is build-specific
# (spconv ships as spconv-cu118 / spconv-cu120 / ...). Everything else is parsed out of
# the requirements-*.txt files, so adding a new optional family needs no edit here.
_UNDECLARABLE_OPTIONAL_MODULES = frozenset({"spconv", "diagnostic_multiring"})


# --------------------------------------------------------------------------------- #
# Source modules
# --------------------------------------------------------------------------------- #

def source_files() -> list[Path]:
    """Every tracked-looking .py file under the source packages."""
    files: list[Path] = []
    for pkg in SOURCE_PACKAGES:
        root = REPO_ROOT / pkg
        if not root.is_dir():
            continue
        files += [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files)


def python_modules() -> list[str]:
    """Dotted module names for every source file, e.g. `watchmal.engine.base_engine`.

    `watchmal` has no `__init__.py` (it is a namespace package), so the dotted name is
    derived from the path rather than from package metadata.
    """
    modules = []
    for path in source_files():
        rel = path.relative_to(REPO_ROOT)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(p.isidentifier() for p in parts):
            continue
        modules.append(".".join(parts))
    return sorted(set(modules))


@lru_cache(maxsize=1)
def optional_modules() -> frozenset[str]:
    """Import names that are optional by design.

    Parsed from the optional requirements files (`requirements-*.txt`); the core
    `requirements.txt` is deliberately not read, since nothing in it may be missing.
    A module that fails to import *only* because one of these is absent is skipped
    rather than failed, which is what lets one suite run in several dependency
    environments.
    """
    names = set(_UNDECLARABLE_OPTIONAL_MODULES)
    for req in sorted(REPO_ROOT.glob("requirements-*.txt")):
        for raw in req.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name = line
            for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";", " "):
                name = name.split(sep)[0]
            name = name.strip().replace("-", "_")
            if name:
                names.add(name)
    return frozenset(names)


def missing_optional_module(exc: BaseException) -> str | None:
    """Name of the optional package behind an import failure, or None.

    Walks the exception chain, because the failure is often wrapped: hydra re-raises a
    `ModuleNotFoundError` from `get_class()` as an `ImportError`, so matching on the
    outer type alone would silently turn "PyG is not installed here" into a test
    failure. Returns None when the missing package is NOT declared optional - that is a
    genuine break and must fail.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name:
            root = current.name.split(".")[0]
            return root if root in optional_modules() else None
        current = current.__cause__ or current.__context__
    return None


def import_or_skip(module_name: str):
    """Import a source module, skipping if an OPTIONAL dependency is what is missing.

    A missing core dependency, a syntax error or a broken relative import still fails:
    only a missing declared-optional package is tolerated.
    """
    import pytest

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        missing = missing_optional_module(exc)
        if missing:
            pytest.skip(f"optional dependency {missing!r} not installed")
        raise


def module_bindings(path: Path) -> set[str]:
    """Names bound at module scope in `path`, by AST - no import, no execution.

    Includes re-exports (`from x import y` makes `y` a valid `_target_` suffix on this
    module, which several configs rely on), assignments and aliases, not just class and
    def statements.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


# --------------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------------- #

def all_config_files() -> list[Path]:
    """Every YAML under tutorial/, whatever tree it belongs to (Tier 0 scope)."""
    tutorial = REPO_ROOT / "tutorial"
    if not tutorial.is_dir():
        return []
    return sorted(p for p in tutorial.rglob("*.y*ml") if p.suffix in (".yaml", ".yml"))


def _entry_files(tree: Path) -> list[Path]:
    """Composable root configs of a tree: `<tree>/main/*.yaml` if that directory exists
    (the caverns layout), else the YAMLs sitting directly in `<tree>` (the watchmal
    layout). Both are supported so neither tree - nor whichever layout survives the
    merge - needs a special case."""
    main = tree / "main"
    if main.is_dir():
        return sorted(main.glob("*.yaml"))
    return sorted(tree.glob("*.yaml"))


def _looks_like_config_tree(directory: Path) -> bool:
    """A config tree has entry configs AND hydra groups (subdirectories of YAML)."""
    if not directory.is_dir():
        return False
    if not _entry_files(directory):
        return False
    return any(
        child.is_dir() and child.name != "main" and any(child.rglob("*.yaml"))
        for child in directory.iterdir()
    )


def config_trees() -> list[Path]:
    """Discover config trees by shape, not by name.

    Today this finds `tutorial/config/{watchmal,caverns}`. When the two trees merge into
    a single one directly under `tutorial/config`, that root matches instead and every
    check keeps working with no edit.
    """
    if not CONFIG_ROOT.is_dir():
        return []
    if _looks_like_config_tree(CONFIG_ROOT):
        return [CONFIG_ROOT]
    return [d for d in sorted(CONFIG_ROOT.iterdir()) if _looks_like_config_tree(d)]


def in_scope_config_trees() -> list[Path]:
    """Config trees Tier 1 gates on (see TIER1_EXCLUDED_CONFIG_TREES)."""
    return [t for t in config_trees() if t.name not in TIER1_EXCLUDED_CONFIG_TREES]


def entry_configs(trees=None) -> list[Path]:
    """Every composable root config of the given trees (in-scope trees by default)."""
    trees = in_scope_config_trees() if trees is None else trees
    return [cfg for tree in trees for cfg in _entry_files(tree)]


def tree_of(entry: Path) -> Path:
    """The config tree an entry config belongs to."""
    return entry.parent.parent if entry.parent.name == "main" else entry.parent


def compose_entry(entry: Path):
    """Compose an entry config the way the launch scripts do.

    `--config-path` points at the directory holding the entry config, and
    `hydra.searchpath` adds the tree root so the group defaults resolve. Composition
    touches no data, no GPU and instantiates nothing - it is pure config resolution.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    tree = tree_of(entry)
    config_dir = entry.parent
    overrides = [] if config_dir == tree else [f"hydra.searchpath=[{tree}]"]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            return compose(config_name=entry.stem, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()


def iter_targets(node, _path=()):
    """Yield `(dotted_target, key_path)` for every `_target_` in a config-ish mapping."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_" and isinstance(value, str):
                yield value, ".".join(_path) or "<root>"
            else:
                yield from iter_targets(value, _path + (str(key),))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from iter_targets(value, _path + (str(i),))


def is_in_repo_target(target: str) -> bool:
    """True for `_target_`s pointing into this repo (as opposed to `torch.optim.Adam`)."""
    return target.split(".")[0] in SOURCE_PACKAGES


def resolve_target_as_text(target: str) -> str | None:
    """Statically check that an in-repo `_target_` names something that exists.

    Returns None when it resolves, or a human-readable reason when it does not. No
    import happens, so this works with zero dependencies installed and reports the
    module path rather than an ImportError traceback.
    """
    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        base = REPO_ROOT / Path(*parts[:split])
        module_file = base.with_suffix(".py")
        if not module_file.is_file():
            module_file = base / "__init__.py"
            if not module_file.is_file():
                continue
        symbol = parts[split]
        if symbol in module_bindings(module_file):
            return None
        return f"module {module_file.relative_to(REPO_ROOT)} does not define or import {symbol!r}"
    return f"no module file found for {target!r}"


# --------------------------------------------------------------------------------- #
# The worker -> engine contract, read out of run.py (never copied)
# --------------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _run_py_tree() -> ast.Module:
    return ast.parse(RUN_PY.read_text(), filename=str(RUN_PY))


def engine_methods_called() -> set[str]:
    """Methods the worker calls on the engine object it knows nothing else about.

    Read live from run.py, so if a step is added to the worker every engine is checked
    for it on the next PR with no test edit.
    """
    return {
        node.func.attr
        for node in ast.walk(_run_py_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "engine"
    }


def engine_ctor_kwargs() -> set[str]:
    """Keyword arguments the worker passes when instantiating the engine.

    Found as the keywords of the `instantiate(config=..., ...)` call; `config` itself is
    hydra's and is not part of the engine signature.
    """
    for node in ast.walk(_run_py_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "instantiate"
        ):
            names = {kw.arg for kw in node.keywords if kw.arg and kw.arg != "config"}
            if names:
                return names
    return set()


def task_keys_consumed_by_worker() -> set[str]:
    """Sub-blocks the worker pops out of a task config before calling the task.

    These are configuration blocks (`data_loaders`, `optimizers`, ...), not arguments of
    the task method, so the conformance check must subtract them before deciding whether
    a task config key is accepted by the engine method.
    """
    return {
        node.args[0].value
        for node in ast.walk(_run_py_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "task_config"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


MAIN_PY = REPO_ROOT / "main.py"

# Name the hydra config is bound to in each entrypoint.
_ENTRYPOINT_CONFIG_NAMES = {MAIN_PY: "config", RUN_PY: "hydra_config"}


def required_top_level_keys() -> dict[str, str]:
    """Top-level config keys the entrypoints read by attribute, as `{key: "file:line"}`.

    Attribute access on a hydra config is *unguarded*: a missing key raises rather than
    returning None, so anything read this way is a hard requirement on every config -
    while anything read through `cfg.get("k", default)` is optional by construction and
    never shows up here.

    One correction is applied: a key that the entrypoint *assigns before reading it*
    is being defaulted, not required (`main.py` generates `seed` when the config omits
    one). A key assigned only after a read - `gpu_list`, normalised from None to [] - is
    still required, because the read comes first and would raise.

    Deriving this instead of listing it means the check is a statement about the
    entrypoints rather than a copy of them: it cannot drift, and a key added to the
    worker is enforced on every config from the next PR onwards.
    """
    loads: dict[str, int] = {}
    assigns: dict[str, int] = {}
    origin: dict[str, str] = {}

    for path, cfg_name in _ENTRYPOINT_CONFIG_NAMES.items():
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        called = {node.func for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == cfg_name
            ):
                continue
            if node in called:  # cfg.get(...) / cfg.items() - a method, not a key
                continue
            where = f"{path.name}:{node.lineno}"
            if isinstance(node.ctx, ast.Load):
                if node.attr not in loads or node.lineno < loads[node.attr]:
                    loads[node.attr] = node.lineno
                    origin[node.attr] = where
            elif isinstance(node.ctx, ast.Store):
                if node.attr not in assigns or node.lineno < assigns[node.attr]:
                    assigns[node.attr] = node.lineno

    return {
        key: origin[key]
        for key, first_read in loads.items()
        if not (key in assigns and assigns[key] < first_read)
    }


# --------------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------------- #

def discover_engines() -> list[tuple[str, type]]:
    """Every concrete engine in the tree, as `(qualified_name, class)`.

    Walks `watchmal.engine`, importing what it can and skipping modules whose optional
    dependency is absent (spconv in a CPU job, PyG in the core job). Subclass-of-
    BaseEngine is the selection criterion, so renames and new families are covered for
    free - which is the point.
    """
    base_module = import_or_skip("watchmal.engine.base_engine")
    base = base_module.BaseEngine

    found: dict[str, type] = {}
    for module_name in python_modules():
        if not module_name.startswith("watchmal.engine"):
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            if missing_optional_module(exc):
                continue
            raise
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, base)
                and attr is not base
                and attr.__module__ == module_name
            ):
                found[f"{attr.__module__}.{attr.__name__}"] = attr
    return sorted(found.items())


def repo_root_on_syspath() -> None:
    """Make `import watchmal` work however pytest was invoked."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
