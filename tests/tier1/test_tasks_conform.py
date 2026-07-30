"""
Tier 1 - the conformance triangle: a config's tasks must be callable on the engine that
same config selects.

Elsewhere Tier 1 checks that configs compose, that `_target_`s resolve and that engines
answer to the worker's API. This is the third edge, and nothing else covers it: that
`tasks:` and `engine:` in one config actually fit each other.

The worker does exactly two things with a task - pop the configuration sub-blocks, then
`getattr(engine, task)(**task_config)` - so a mismatch is a `TypeError` raised after the
model is built and the data is loaded, i.e. minutes into a cluster job, or an
`AttributeError` on a task name no engine implements.

Why this matters more than it looks: the task groups and the engine signatures have
already diverged between the two families. `train()` takes `num_val_batches` on the
image side and does not on the graph side; the inference task is `evaluate` for graph
engines and `test` for multi-ring. Merging the two config trees is the step that brings
those groups into contact - and this check turns that class of breakage into a red PR in
seconds, naming the config and the rejected parameter, instead of a crash on a node.

Both sides are discovered - the configs by globbing, the engine by reading its own
`_target_`, the consumed sub-blocks by AST from run.py - so a rename, a new engine
family or a merged config tree changes nothing here.
"""

import inspect

import pytest
from omegaconf import OmegaConf

from tests.discovery import (
    REPO_ROOT,
    compose_entry,
    entry_configs,
    missing_optional_module,
    task_keys_consumed_by_worker,
)

ENTRY_CONFIGS = entry_configs()
ENTRY_IDS = [str(p.relative_to(REPO_ROOT)) for p in ENTRY_CONFIGS]

# Sub-blocks run.py pops out of a task config before calling the task: they configure
# the engine, they are not arguments of the task method.
CONSUMED = task_keys_consumed_by_worker()


def test_consumed_blocks_were_extracted_from_the_entrypoint():
    assert CONSUMED, "no task_config.pop(...) found in run.py - the AST reader is stale"


def _engine_class(config, entry):
    from hydra.utils import get_class

    engine_config = config.get("engine")
    assert engine_config is not None and "_target_" in engine_config, (
        f"{entry.relative_to(REPO_ROOT)} has no engine._target_"
    )
    try:
        return get_class(engine_config._target_)
    except ImportError as exc:
        missing = missing_optional_module(exc)
        if missing:
            pytest.skip(f"optional dependency {missing!r} not installed")
        raise


@pytest.mark.parametrize("entry", ENTRY_CONFIGS, ids=ENTRY_IDS)
def test_tasks_are_callable_on_the_selected_engine(entry):
    config = compose_entry(entry)
    engine_cls = _engine_class(config, entry)

    tasks = config.get("tasks") or {}
    assert tasks, f"{entry.relative_to(REPO_ROOT)} defines no tasks to run"

    for task_name in tasks:
        method = getattr(engine_cls, task_name, None)
        assert callable(method), (
            f"{entry.relative_to(REPO_ROOT)}: task '{task_name}' is not a method of "
            f"{engine_cls.__module__}.{engine_cls.__name__}. run.py dispatches tasks "
            "with getattr(engine, task), so this config cannot run."
        )


@pytest.mark.parametrize("entry", ENTRY_CONFIGS, ids=ENTRY_IDS)
def test_task_parameters_are_accepted_by_the_engine(entry):
    config = compose_entry(entry)
    engine_cls = _engine_class(config, entry)

    tasks = config.get("tasks") or {}
    for task_name, task_config in tasks.items():
        method = getattr(engine_cls, task_name, None)
        if not callable(method):
            continue  # reported by test_tasks_are_callable_on_the_selected_engine
        block = OmegaConf.to_container(task_config, resolve=False) if task_config else {}
        if not isinstance(block, dict):
            continue

        parameters = inspect.signature(method).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue  # **kwargs accepts anything

        unknown = sorted(set(block) - set(parameters) - CONSUMED)
        assert not unknown, (
            f"{entry.relative_to(REPO_ROOT)}: tasks.{task_name} passes {unknown}, which "
            f"{engine_cls.__name__}.{task_name}() does not accept "
            f"(accepts: {sorted(set(parameters) - {'self'})}; consumed by the worker: "
            f"{sorted(CONSUMED)}). This config would raise TypeError at dispatch."
        )
