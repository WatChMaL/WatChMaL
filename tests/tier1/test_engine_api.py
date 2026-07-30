"""
Tier 1 - every engine satisfies the worker -> engine contract.

Goal: make "one single core" a property the repo proves rather than a claim in a
document. `watchmal/entrypoints/run.py` is now the only worker: it builds a model,
instantiates an engine it knows nothing about, calls a fixed set of configuration
methods on it and then dispatches the tasks. Nothing enforces that an engine actually
answers to that interface - the contract holds by convention, and would break the moment
someone adds an engine family or renames a method on the base while a subclass still
overrides the old name.

The expectations are read out of `run.py` by AST rather than listed here, so this file
is a statement *about* the entrypoint instead of a second copy of it. If a step is added
to the worker, every engine is checked against it on the next PR with no edit to this
test; if a method is renamed, the check follows the rename automatically.

The subjects are discovered the same way: any subclass of `BaseEngine` anywhere under
`watchmal.engine`, which is exactly the population the upcoming class-renaming step will
churn, and it covers engines that do not exist yet.
"""

import inspect

import pytest

from tests.discovery import (
    discover_engines,
    engine_ctor_kwargs,
    engine_methods_called,
    import_or_skip,
)

ENGINES = discover_engines()
ENGINE_IDS = [name for name, _cls in ENGINES]
REQUIRED_METHODS = sorted(engine_methods_called())
REQUIRED_CTOR_KWARGS = sorted(engine_ctor_kwargs())


def test_contract_was_extracted_from_the_entrypoint():
    """If the AST reader stops finding the calls, every check below passes vacuously."""
    assert REQUIRED_METHODS, "no engine method call found in run.py - reader is stale"
    assert REQUIRED_CTOR_KWARGS, "no engine instantiate() kwargs found in run.py"


def test_engines_are_discovered():
    assert ENGINES, "no BaseEngine subclass found under watchmal.engine"


@pytest.mark.parametrize("engine_cls", [cls for _n, cls in ENGINES], ids=ENGINE_IDS)
def test_engine_inherits_the_single_base(engine_cls):
    base = import_or_skip("watchmal.engine.base_engine").BaseEngine
    assert issubclass(engine_cls, base)


@pytest.mark.parametrize("engine_cls", [cls for _n, cls in ENGINES], ids=ENGINE_IDS)
def test_engine_implements_methods_the_worker_calls(engine_cls):
    missing = [name for name in REQUIRED_METHODS if not callable(getattr(engine_cls, name, None))]
    assert not missing, (
        f"{engine_cls.__module__}.{engine_cls.__name__} does not implement {missing}, "
        f"which run.py calls on every engine ({REQUIRED_METHODS})."
    )


@pytest.mark.parametrize("engine_cls", [cls for _n, cls in ENGINES], ids=ENGINE_IDS)
def test_engine_ctor_accepts_what_the_worker_passes(engine_cls):
    """run.py instantiates every engine with the same keyword arguments.

    An engine may add its own (the multi-ring engine takes `train_output_dir`, the graph
    classifier takes `prediction_threshold`) - those come from the config. It may not
    *drop* one of the shared ones, because the worker passes them unconditionally.
    """
    signature = inspect.signature(engine_cls.__init__)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_kwargs:
        return
    missing = [name for name in REQUIRED_CTOR_KWARGS if name not in signature.parameters]
    assert not missing, (
        f"{engine_cls.__module__}.{engine_cls.__name__}.__init__ does not accept "
        f"{missing}, which run.py passes to instantiate() for every engine "
        f"({REQUIRED_CTOR_KWARGS})."
    )
