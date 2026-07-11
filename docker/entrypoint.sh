#!/usr/bin/env bash
# Activate the conda `pixie` env, then exec whatever command was passed
# (defaults to an interactive bash shell via the Dockerfile CMD).
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate pixie
exec "$@"
