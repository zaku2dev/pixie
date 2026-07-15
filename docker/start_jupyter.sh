#!/usr/bin/env bash
# Launch JupyterLab inside the container, bound so it is reachable from the host
# (SSH tunnel or platform HTTP proxy). The pixie env's jupyter is used, so the
# default kernel is the pixie interpreter.
#
# Usage (inside the container):
#   start_jupyter.sh                 # port 8888, token auth (prints the URL)
#   PORT=9999 start_jupyter.sh       # different port
#   JUPYTER_TOKEN=mysecret start_jupyter.sh   # fixed token instead of a random one
#
# Connect from your local machine (works on any SSH-capable host):
#   ssh -N -L 8888:localhost:8888 <user>@<host> -p <ssh_port>
# then open the http://localhost:8888/lab?token=... URL this script prints.
set -euo pipefail

# Ensure the pixie env is active even if this is run from a non-login shell.
if [[ "${CONDA_DEFAULT_ENV:-}" != "pixie" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pixie
fi

PORT="${PORT:-8888}"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-/opt/pixie}"

exec jupyter lab \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --no-browser \
    --allow-root \
    --notebook-dir="${NOTEBOOK_DIR}"
