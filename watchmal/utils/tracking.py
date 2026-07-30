"""
Unified run tracking for the single WatChMaL core.

One `RunTracker` owns both tracking backends so every engine gets the same options,
regardless of which model family it belongs to:

  * CSV  (default on) - writes `outputs/log_train_{rank}.csv` and `outputs/log_val.csv`
    in exactly the schema `analysis/read.py` parses (per-step train rows
    `iteration,epoch,<metrics>`; per-validation rows `...,saved_best`). This is what keeps
    the graph / multi-ring runs readable by the existing `analysis/` code, which before
    the merge only understood the CNN core's CSVs.
  * wandb (on iff a wandb run is active) - thin guarded passthroughs so call sites never
    have to branch on `wandb_run is not None`, and a CSV-only run never imports wandb.

Rank policy matches the pre-merge CNN core: the train CSV is written per rank
(`log_train_{rank}.csv`), the validation CSV and all wandb logging happen on rank 0 only.
`analysis/read.py` globs `log_train_*.csv`, so a single rank-0 file (the graph/MR case,
which only accumulates metrics on rank 0) parses fine too.
"""

from watchmal.utils.logging_utils import CSVLog, setup_logging

log = setup_logging(__name__)


class RunTracker:
    def __init__(self, dump_path, rank, wandb_run=None, csv_enabled=True):
        """
        Parameters
        ----------
        dump_path : str
            Output directory for the CSV files (a trailing '/' is added if missing).
        rank : int
            This process' rank; drives the CSV filename and the rank-0 gating.
        wandb_run : optional
            An active wandb run, or None. The caller is expected to pass None on
            non-zero ranks (as the entrypoints already do).
        csv_enabled : bool
            Whether to write the analysis-compatible CSV logs. wandb is independently
            enabled by whether `wandb_run` is not None. CSV stays an option this way.
        """
        self.dump_path = dump_path if dump_path.endswith("/") else dump_path + "/"
        self.rank = rank
        self.wandb_run = wandb_run
        self.csv_enabled = csv_enabled
        self._train_log = None  # lazily created CSVLog (per rank)
        self._val_log = None    # lazily created CSVLog (rank 0)

    # ------------------------------------------------------------------ #
    # wandb passthroughs - no-ops when no run is active
    # ------------------------------------------------------------------ #
    @property
    def has_wandb(self):
        return self.wandb_run is not None

    def wandb_log(self, data, **kwargs):
        if self.wandb_run is not None:
            self.wandb_run.log(data, **kwargs)

    def wandb_watch(self, *args, **kwargs):
        if self.wandb_run is not None:
            self.wandb_run.watch(*args, **kwargs)

    def wandb_save(self, path):
        if self.wandb_run is not None:
            self.wandb_run.save(path)

    # ------------------------------------------------------------------ #
    # analysis-compatible CSV
    # ------------------------------------------------------------------ #
    def train_step(self, iteration, epoch, metrics):
        """Append one training-step row to `log_train_{rank}.csv`.

        `metrics` is a dict of python scalars (e.g. {'loss': ..., 'accuracy': ...}).
        Keys must be stable across steps (the header is taken from the first row).
        """
        if not self.csv_enabled:
            return
        if self._train_log is None:
            self._train_log = CSVLog(f"{self.dump_path}log_train_{self.rank}.csv")
        row = {"iteration": int(iteration), "epoch": int(epoch)}
        row.update({k: float(v) for k, v in metrics.items()})
        self._train_log.log(row)

    def validation(self, iteration, epoch, metrics, saved_best):
        """Append one validation row to `log_val.csv` (rank 0 only)."""
        if not self.csv_enabled or self.rank != 0:
            return
        if self._val_log is None:
            self._val_log = CSVLog(f"{self.dump_path}log_val.csv")
        row = {"iteration": int(iteration), "epoch": int(epoch)}
        row.update({k: float(v) for k, v in metrics.items()})
        row["saved_best"] = bool(saved_best)
        self._val_log.log(row)

    def close(self):
        if self._train_log is not None:
            self._train_log.close()
            self._train_log = None
        if self._val_log is not None:
            self._val_log.close()
            self._val_log = None
