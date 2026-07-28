# Checks

Two tiers, one per GitHub workflow. Both are runnable locally and are meant to be — they
are the pre-flight check before a cluster submission, not just CI decoration.

| | What it answers | Needs | Cost | Workflow |
|---|---|---|---|---|
| **Tier 0** `tests/tier0` | Can this revision be read, parsed and resolved as text? | `pytest pyyaml ruff` | ~30 s | `.github/workflows/tier0-static.yml` |
| **Tier 1** `tests/tier1` | Would it get past start-up on a cluster? | the framework installed | ~4 min | `.github/workflows/tier1-contract.yml` |

Each workflow file opens with a full statement of its goal, what it protects and what it
deliberately does not cover. Read those first.

```bash
pytest tests/tier0                        # no framework needed
pytest tests/tier1                        # needs requirements.txt installed
ruff check watchmal analysis main.py      # same ruleset CI uses (pyproject.toml)
pytest -m transitional --collect-only -q  # what is left to unify
```

## Design rule

> A check asserts a property of the target architecture, **discovers** its subjects at
> runtime, and routes through the entrypoint-facing API rather than family-specific
> internals.

Everything the core unification still has to change — engine class names, the two config
trees becoming one, retiring the shim modules — is a *name* or a *location*, never a
*behaviour*. So no check here hardcodes an inventory: modules are found by walking the
package, configs by globbing, config trees by shape, engines as `BaseEngine` subclasses,
and the worker→engine contract by reading `watchmal/entrypoints/run.py` with `ast`. A
check written this way cannot be broken by a rename, and it covers engines and configs
that do not exist yet. `tests/discovery.py` is where all of that lives.

## Two mechanisms worth knowing

**The ledger** — `tests/data/known_broken_targets.txt` lists `_target_`s that do not
resolve today (three stale references inherited from upstream). A *new* broken target
fails the build; *fixing* a listed one also fails the build, because the expectation is a
strict xfail — so the ledger line gets deleted in the same PR. Empty file ⇒ the check
becomes a permanent guard with no code change.

**`@pytest.mark.transitional`** — a check that is only true mid-merge, carrying a
module-level `RETIRES_WITH` naming the step that kills it. `pytest -m transitional
--collect-only -q` prints the remaining unification work, generated from the suite rather
than maintained by hand. Where possible these are parametrised over the offending
subjects, so the last fix empties the parametrisation and the check disappears on its own.

## Scope

Tier 1's hydra checks currently gate on the caverns config tree only; the ex-watchmal tree
is excluded via `TIER1_EXCLUDED_CONFIG_TREES` in `tests/discovery.py` — the suite's single
hardcoded config name, and a deliberate, temporary decision. Tier 0 still parses every
YAML in it. `test_excluded_config_trees_still_exist` fails the day that tree disappears,
which is the signal to delete the constant and gate the merged tree for free.
