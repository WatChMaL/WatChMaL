import subprocess
import warnings


class CSVData:
    """
    Class to organize output csv file
    """
    def __init__(self,fout):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}

    def record(self, input_dict):
        self._dict = input_dict.copy()

    def write(self):
        if self._str is None:
            self._fout=open(self.name,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()


def get_git_version(path):
    try:
        git_version = subprocess.check_output(['git', '-C', path, 'describe', '--always', '--long', '--tags', '--dirty'], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        if b"not a git repository" in e.stderr:
            warnings.warn("WARNING: Path is not in a git repository so version tracking is not available.", stacklevel=2)
        else:
            warnings.warn("WARNING: Error when attempting to check git version so version tracking is not available.", stacklevel=2)
        return None
    else:
        git_version = git_version.decode().strip()
        if "-dirty" in git_version:
            warnings.warn("WARNING: The git repository has uncommitted changes. Please commit changes before running"
                          " WatChMaL code for proper version control", stacklevel=2)
        return git_version
