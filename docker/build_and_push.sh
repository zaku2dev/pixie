#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build the Pixie image on a GPU pod and push it to a registry (e.g. Docker Hub)
# so future RunPod instances can start from it instantly.
#
# Choose the target GPU as the first argument — each preset builds a SINGLE
# CUDA arch (fastest build, smallest image) and tags a distinct image so both
# can live in the registry side by side:
#
#   IMAGE_REPO=youruser/pixie ./docker/build_and_push.sh 4090
#       -> RTX 4090 / L40 (sm_89)  -> youruser/pixie:sm89-4090
#
#   IMAGE_REPO=youruser/pixie ./docker/build_and_push.sh a6000
#       -> A6000 / A40 / A5000 / 3090 (sm_86) -> youruser/pixie:sm86-a6000
#
# Env overrides:
#   IMAGE_REPO   registry repo WITHOUT a tag (required), e.g. youruser/pixie
#   IMAGE        full name:tag (optional) — overrides the auto-generated tag
#   ARCH         CUDA arch list (optional) — overrides the preset default,
#                e.g. ARCH="8.0;8.6;8.9;9.0" for one image that runs anywhere
#   MAX_JOBS     flash-attn parallel compile jobs (default 6, safe on a 64 GB
#                box; raise on a bigger host for a faster build)
#   GITHUB_TOKEN GitHub PAT with read access to the (private) pixie repo. The
#                build clones the code over HTTPS using this token, passed as a
#                BuildKit secret so it is never baked into an image layer.
#   PIXIE_REF    branch/tag to clone (default: the Dockerfile's PIXIE_REF).
# -----------------------------------------------------------------------------
set -euo pipefail

# The code is git-cloned inside the image from a private repo, so a token is
# required at build time. BuildKit is required to consume the --secret mount.
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN=<github PAT with repo read access> (used to clone the private repo at build time)}"
export DOCKER_BUILDKIT=1

GPU="${1:-${GPU:-}}"

case "$GPU" in
  4090|l40|ada)
    ARCH_DEFAULT="8.9"; TAG="sm89-4090" ;;
  a6000|a40|a5000|3090|ampere)
    ARCH_DEFAULT="8.6"; TAG="sm86-a6000" ;;
  "")
    echo "Usage: IMAGE_REPO=<user>/pixie $0 <4090|a6000>" >&2
    echo "  4090   RTX 4090 / L40            (sm_89)" >&2
    echo "  a6000  A6000 / A40 / A5000 / 3090 (sm_86)" >&2
    exit 1 ;;
  *)
    echo "Unknown GPU preset '$GPU'. Use '4090' (sm_89) or 'a6000' (sm_86)." >&2
    exit 1 ;;
esac

# ARCH env var (e.g. a multi-arch list) overrides the preset default.
ARCH="${ARCH:-$ARCH_DEFAULT}"
MAX_JOBS="${MAX_JOBS:-6}"

# Resolve the image name: explicit IMAGE wins, else IMAGE_REPO + preset tag.
if [[ -z "${IMAGE:-}" ]]; then
  IMAGE_REPO="${IMAGE_REPO:?Set IMAGE_REPO=<registry-user>/pixie (repo without a tag)}"
  IMAGE="${IMAGE_REPO}:${TAG}"
fi

# Run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

# Cloud GPUs are x86-64, so always target linux/amd64. Build on a NATIVE
# x86-64 Linux host: on Apple Silicon this forces slow, crash-prone emulation
# (the CUDA/conda binaries fail under Rosetta) — see DOCKER_RUNPOD.md Step 1.
PLATFORM="${PLATFORM:-linux/amd64}"

echo ">> Building ${IMAGE} for GPU=${GPU} (platform=${PLATFORM}, TORCH_CUDA_ARCH_LIST=${ARCH})"
docker build \
  --platform "${PLATFORM}" \
  --secret id=github_token,env=GITHUB_TOKEN \
  ${PIXIE_REF:+--build-arg PIXIE_REF="${PIXIE_REF}"} \
  --build-arg TORCH_CUDA_ARCH_LIST="${ARCH}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE}" \
  .

echo ">> Pushing ${IMAGE}"
docker push "${IMAGE}"

echo ">> Done: ${IMAGE}"
