#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build the Pixie image on a GPU pod and push it to a registry (e.g. Docker Hub)
# so future RunPod instances can start from it instantly.
#
# Usage:
#   IMAGE=yourdockerhubuser/pixie:latest ./docker/build_and_push.sh
#   IMAGE=yourdockerhubuser/pixie:latest ARCH=8.6 ./docker/build_and_push.sh
# -----------------------------------------------------------------------------
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE=<registry-user>/pixie:tag}"
ARCH="${ARCH:-8.0;8.6;8.9;9.0}"   # override with your GPU's compute capability
MAX_JOBS="${MAX_JOBS:-16}"

# Run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

echo ">> Building ${IMAGE} for TORCH_CUDA_ARCH_LIST=${ARCH}"
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="${ARCH}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE}" \
  .

echo ">> Pushing ${IMAGE}"
docker push "${IMAGE}"

echo ">> Done: ${IMAGE}"
