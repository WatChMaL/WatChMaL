"""
Tier 1 - every module imports, and the optional-dependency boundary holds.

Two goals, and the second is the reason this job is run in more than one environment.

1. Import-smoke. The merge re-parented engines onto a single base, moved methods between
   classes and deleted two entrypoints; `compileall` proves none of that is a syntax
   error, but only an import proves the module graph still resolves. Parametrising per
   module (rather than importing the package once) means a failure names the module.

2. The dependency boundary. `wandb`, `torch_geometric` and `spconv` are optional by
   design: CSV tracking must work with no wandb installed, and an image or multi-ring
   run must not need PyG. That is a property of the *code*, so it can only be tested by
   running in an environment where those packages are genuinely absent - which is what
   the `core` CI job is for. It deliberately installs requirements.txt and nothing else;
   if it goes red, the boundary has been breached and the fix is in the source, never in
   the workflow.

A module that fails to import *only* because an optional package is missing is skipped,
not failed - that is what lets the same suite run in the core and graph environments.
The optional set is parsed from the requirements-*.txt files, so declaring a new
optional family needs no change here.
"""

import subprocess
import sys
import textwrap

import pytest

from tests.discovery import (
    REPO_ROOT,
    discover_engines,
    import_or_skip,
    optional_modules,
    python_modules,
)

MODULES = python_modules()


def test_modules_are_discovered():
    assert MODULES, "no source module found - have the packages moved?"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    import_or_skip(module_name)


def _import_in_subprocess(statements: str) -> set[str]:
    """Run `statements` in a fresh interpreter, return the optional modules it loaded.

    A subprocess is what makes this honest: inside the pytest process another test may
    already have imported PyG, so `sys.modules` there proves nothing.
    """
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        {statements}
        print(json.dumps(sorted(sys.modules)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    import json

    loaded = {name.split(".")[0] for name in json.loads(result.stdout.splitlines()[-1])}
    return loaded & set(optional_modules())


def test_entrypoint_pulls_no_optional_dependency():
    """`import main` must not drag in wandb, PyG or spconv.

    This is the falsification test for the lazy-import work: wandb is imported inside
    the branch that actually starts a run, and the graph/multi-ring code is reached only
    through hydra `_target_`s. Checked against the single entrypoint, so it keeps
    holding however the families are reorganised behind it.
    """
    leaked = _import_in_subprocess("import main")
    assert not leaked, (
        f"`import main` loaded optional dependencies {sorted(leaked)}. They must be "
        "imported lazily, at the point of use, so a run that does not need them works "
        "without them installed."
    )


def test_some_engine_is_importable_with_core_requirements_only():
    """At least one engine must import in whatever environment this runs in.

    In the `core` CI job - requirements.txt only - this is the canary for an optional
    import leaking into shared code: when `dataset/data_utils.py` imported PyG at module
    scope, it did so on the CNN engine's import path, and every engine in the repo
    became unimportable without a package that is in no requirements file. Stated over
    the discovered engine set rather than a named class, so it survives P5's renames.
    """
    engines = discover_engines()
    assert engines, (
        "no engine could be imported. If this is the core job, an optional dependency "
        "(torch_geometric / spconv / wandb) has leaked into a shared module - check "
        "which import fails in the parametrised test_module_imports results above."
    )
