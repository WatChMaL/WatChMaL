#!/bin/bash

# Create your personal workspace from the shipped tutorial tree.
# Run from the repo root:  bash setup/make_dirs.sh
# The copies are meant to be your local scratch; the tracked tutorial/ tree stays the
# reference. Copies whichever of config/launch exist, so it also works mid-merge while
# the caverns and launch trees are still being added.

# Check we are at the repo root
if [[ ! -d "tutorial" ]]; then
    echo "Error: run this from the WatChMaL root directory ('tutorial/' not found)"
    exit 1
fi

echo "Welcome to WatChMaL!"

mkdir -p config launch
# `src/.` copies the *contents* of src (portable across GNU/BSD cp, no nesting)
[[ -d tutorial/config ]] && cp -R tutorial/config/. config/
[[ -d tutorial/launch ]] && cp -R tutorial/launch/. launch/

echo "Copied tutorial/config -> config/ and tutorial/launch -> launch/ (whichever exist)."
echo "Customize these freely; keep the tracked tutorial/ tree as the reference."
