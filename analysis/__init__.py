import os
from WatChMaL.watchmal.utils.logging_utils import get_git_version

print(f"Imported analysis code from WatChMaL repository with git version: {get_git_version(os.path.dirname(__file__))}")