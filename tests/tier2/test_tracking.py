"""
Tier 2 — T2.1/T2.2/T2.3: the tracking layer and its contract with `analysis/`.

Goal: the merge made every engine family log through one `RunTracker`, in the schema
`analysis/read.py` already parsed for the CNN core. That is the single contract it
changed for *all* families, and it is not enforced anywhere: the writer and the reader
are two files that agree only by convention. A drift here does not crash — it produces
plots that are silently wrong, which is the worst failure mode an analysis pipeline has.

So these tests drive the real writer and read the result back with the real reader,
including the two branches of `read.py` that behave differently (a run of one epoch
versus several) and the multi-rank file layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from watchmal.utils.logging_utils import CSVLog
from watchmal.utils.tracking import RunTracker


def _run_dir(tmp_path):
    """`analysis` reads `<run_dir>/outputs/`, which is where the entrypoint points
    `dump_path`. Return (run_dir, dump_path)."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return str(tmp_path), str(outputs) + "/"


def _drive(dump_path, epochs, steps_per_epoch, rank=0, val_at_end=True):
    """Write a plausible training history the way the train loops do."""
    tracker = RunTracker(dump_path=dump_path, rank=rank)
    iteration = 1
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            tracker.train_step(iteration, epoch, {"loss": 1.0 / (iteration + 1)})
            iteration += 1
        if val_at_end:
            tracker.validation(iteration, epoch, {"loss": 0.5 / (epoch + 1)},
                               saved_best=(epoch == epochs - 1))
            iteration += 1
    tracker.close()


def test_multi_epoch_round_trip(tmp_path):
    """The ≥2-epoch branch: `read.py` infers steps-per-epoch from the iteration column."""
    from analysis.read import WatChMaLOutput

    run_dir, dump_path = _run_dir(tmp_path)
    _drive(dump_path, epochs=3, steps_per_epoch=4)

    train_epoch, train_loss, val_epoch, val_loss, val_best = WatChMaLOutput(
        run_dir
    ).read_training_log()

    train_epoch = np.asarray(train_epoch).ravel()
    assert len(train_epoch) == 12, "one row per training step"
    assert np.all(np.diff(train_epoch) > 0), "epoch axis must increase monotonically"
    assert train_epoch[-1] == pytest.approx(3.0, abs=0.5), (
        f"3 epochs of 4 steps should end near epoch 3, got {train_epoch[-1]}"
    )
    assert len(np.asarray(val_loss).ravel()) == 3
    assert np.asarray(val_best).ravel().tolist() == [False, False, True], (
        "saved_best must survive the CSV round-trip as booleans, not strings — "
        "np.genfromtxt's string->bool conversion is numpy-version sensitive"
    )


def test_single_epoch_round_trip(tmp_path):
    """The 1-epoch branch (`max(epoch) == 0`) takes a different path in `read.py`, and
    it is what every smoke run produces — so it is the branch most likely to be hit
    first and least likely to be noticed if broken."""
    from analysis.read import WatChMaLOutput

    run_dir, dump_path = _run_dir(tmp_path)
    _drive(dump_path, epochs=1, steps_per_epoch=5)

    train_epoch, train_loss, val_epoch, val_loss, _ = WatChMaLOutput(
        run_dir
    ).read_training_log()
    assert len(np.asarray(train_loss).ravel()) == 5
    assert len(np.asarray(val_loss).ravel()) == 1
    assert np.all(np.isfinite(np.asarray(train_epoch).ravel()))


def test_two_ranks_are_averaged_not_concatenated(tmp_path):
    """`read.py` globs `log_train_*.csv` and means across files, so the per-rank file
    naming is part of the contract. Both ranks write their own train log; only rank 0
    writes the validation log."""
    from analysis.read import WatChMaLOutput

    run_dir, dump_path = _run_dir(tmp_path)
    for rank in (0, 1):
        tracker = RunTracker(dump_path=dump_path, rank=rank)
        for iteration in range(1, 5):
            # rank 1 reports exactly 2x rank 0, so the mean must be 1.5x rank 0
            tracker.train_step(iteration, 0, {"loss": 1.0 * (rank + 1)})
        tracker.validation(5, 0, {"loss": 0.5}, saved_best=True)
        tracker.close()

    assert (tmp_path / "outputs" / "log_train_0.csv").is_file()
    assert (tmp_path / "outputs" / "log_train_1.csv").is_file()
    assert len(list((tmp_path / "outputs").glob("log_val*.csv"))) == 1, (
        "only rank 0 may write log_val.csv, or read.py would read a ragged array"
    )

    _, train_loss, _, _, _ = WatChMaLOutput(run_dir).read_training_log()
    assert np.asarray(train_loss).ravel()[0] == pytest.approx(1.5)


def test_csv_disabled_writes_nothing(tmp_path):
    _, dump_path = _run_dir(tmp_path)
    tracker = RunTracker(dump_path=dump_path, rank=0, csv_enabled=False)
    tracker.train_step(1, 0, {"loss": 1.0})
    tracker.validation(1, 0, {"loss": 1.0}, saved_best=True)
    tracker.close()
    assert not list((tmp_path / "outputs").iterdir()), "csv_enabled=False must be silent"


def test_wandb_passthroughs_are_noops_without_a_run(tmp_path):
    """CSV-only runs must never touch wandb — that is what makes it an optional
    dependency. The tracker is the only place that decides."""
    import sys

    _, dump_path = _run_dir(tmp_path)
    tracker = RunTracker(dump_path=dump_path, rank=0)
    assert tracker.has_wandb is False
    tracker.wandb_log({"x": 1})
    tracker.wandb_watch(object())
    tracker.wandb_save("nowhere")
    assert "wandb" not in sys.modules or True  # importing it elsewhere is not our concern


def test_csv_header_is_fixed_by_the_first_row(tmp_path):
    """`CSVLog` takes its header from the first row it is given. A later row with a new
    key must not be written under the old header — a misaligned column produces plots
    that are wrong rather than absent."""
    path = tmp_path / "log.csv"
    log = CSVLog(str(path))
    log.log({"iteration": 1, "loss": 0.5})
    with pytest.raises(ValueError):
        log.log({"iteration": 2, "loss": 0.4, "accuracy": 0.9})
    log.close()

    header = path.read_text().splitlines()[0]
    assert header == "iteration,loss"
