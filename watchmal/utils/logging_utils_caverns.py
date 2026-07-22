"""
Logging helpers for the CAVERNS core.

The watchmal core has its own copy in logging_utils_watchmal.py. CSVLog and
get_git_version are identical in both; they are duplicated rather than shared so
neither core depends on the other's file.
"""

import subprocess
import csv
import logging

# get_git_version() below logs through this. Upstream defines it too; it was dropped
# when this file was forked, which left those warning branches raising NameError.
log = logging.getLogger(__name__)


def setup_logging(name: str):
    """Return a plain module logger and let the logging config own the handlers.

    Under Hydra, records propagate up to the root logger, whose handlers write
    BOTH the console output and <run_dir>/main.log (hydra job_logging config).
    Do not attach a StreamHandler here or set propagate=False: the first
    duplicates every console line, the second cuts the loggers off from the
    root logger and leaves main.log empty.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the minimum logging level
    return logger


class DisplayFilter(logging.Filter):
    """A custom filter which enables or disables logging based on a flag."""
    def __init__(self, display=True):
        super().__init__()
        self.display = display

    def filter(self, record):
        """Determines if the specified record is to be logged."""
        return self.display

def setup_logging_with_filter(name: str):
    logger = setup_logging(name)

    # Logger-level filter: applied before propagation, so it also gates what
    # reaches the root logger's console/main.log handlers.
    display_filter = DisplayFilter()
    logger.addFilter(display_filter)

    # Function to change display setting dynamically
    def set_display(display):
        display_filter.display = display

    # Attach the function to the logger object
    logger.set_display = set_display

    return logger

class CSVLog:
    """
    Class to organize output csv file
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None

    def log(self, fields):
        if self.file is None:
            self.file = open(self.filename, 'w', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=fields.keys())
            self.writer.writeheader()
        self.writer.writerow(fields)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_git_version(path):
    try:
        git_version = subprocess.check_output(['git', '-C', path, 'describe', '--always', '--long', '--tags', '--dirty'], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        if b"not a git repository" in e.stderr:
            log.warning("WARNING: Path is not in a git repository so version tracking is not available.", stacklevel=2)
        else:
            log.warning("WARNING: Error when attempting to check git version so version tracking is not available.", stacklevel=2)
        return None
    else:
        git_version = git_version.decode().strip()
        if "-dirty" in git_version:
            log.warning("WARNING: The git repository has uncommitted changes. Please commit changes before running "
                        "WatChMaL code for proper version control", stacklevel=2)
        return git_version
