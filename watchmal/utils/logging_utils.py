import subprocess
import csv
import logging

log = logging.getLogger(__name__)


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
