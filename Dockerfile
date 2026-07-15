# =============================================================================
# Pixie — GPU Docker image (conda-based)
# =============================================================================
# Reproduces the full README.md + environment.yaml setup so a fresh GPU instance
# (e.g. RunPod) can run Pixie without redoing the install by hand.
#
# BUILD (needs a fast x86-64 host with lots of RAM; a GPU is NOT required — the
# CUDA arch is baked in via the build args below):
#     docker build -t <dockerhub-user>/pixie:latest .
#
# Defaults target the A6000 (sm_86) with MAX_JOBS=6, which fits a 64 GB box.
# Override to build for a different card or a wider set of GPUs, e.g.:
#     docker build \
#       --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \  # run on many cards
#       --build-arg MAX_JOBS=16 \                             # bigger box, faster
#       -t <dockerhub-user>/pixie:latest .
#
# The base -devel image ships nvcc, which is required to compile
# tiny-cuda-nn, diff-gaussian-rasterization and PyTorch3D (flash-attn ships as a
# prebuilt wheel, but its CUDA runtime still expects this -devel base).
# See DOCKER_RUNPOD.md / DOCKER_VASTAI.md for the full build/push/run walkthroughs.
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---- Build-time knobs -------------------------------------------------------
# GPU compute capabilities to compile CUDA extensions for. `docker build` on
# RunPod usually cannot see a GPU, so auto-detection fails — pin the arch(es)
# explicitly. Defaults cover the common RunPod cards:
#   8.0 A100 · 8.6 A6000/A40/A10/3090 · 8.9 4090/L40 · 9.0 H100
# flash-attn requires sm80+, so keep the floor at 8.0. Defaults to a single arch
# (8.6, A6000/A40/A10/3090) for a fast, low-memory build. Override with a wider
# list (e.g. "8.0;8.6;8.9;9.0") if you need one image to run on other cards.
ARG TORCH_CUDA_ARCH_LIST="8.6"
# Parallel compile jobs for the source-built CUDA extensions (tiny-cuda-nn,
# PyTorch3D, the gaussian rasterizer, simple-knn). flash-attn — historically the
# memory hog this knob existed for — now installs as a prebuilt wheel (STEP 7),
# so peak RAM is far lower. 6 suits a 64 GB box; use 4 on 32 GB, 2 on 16 GB.
# Raise it on a bigger box for a faster build, lower it if the build OOMs.
ARG MAX_JOBS=6
# Blender version (README pins 4.3.2).
ARG BLENDER_SERIES=4.3
ARG BLENDER_VERSION=4.3.2
# Blender add-ons, pinned for reproducibility (override if you use a fork).
# BlenderNeRF: git tag; its GitHub releases ship no asset zip, so we fetch the
# source archive of the tag. NOTE: config/paths/default.yaml defaults to a
# *custom* "BlenderNeRF-main-custom.zip" — if the paper used a fork, override
# BLENDER_NERF_REF or bind-mount your own zip and repoint the config.
ARG BLENDER_NERF_REF=v6
# gaussian-splatting-blender-addon: no releases exist, so pin a commit SHA.
ARG BLENDER_GS_ADDON_REF=dad654521f5a8d091050219b756ada93d90da98f

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    PATH=/opt/conda/bin:/opt/blender:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ---- System dependencies ----------------------------------------------------
# build tooling + OpenGL/X libs for headless Blender rendering (xvfb) and the
# viser/nerfstudio viewers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git git-lfs wget curl ca-certificates \
        ninja-build cmake pkg-config xz-utils unzip \
        libgl1 libglu1-mesa libglib2.0-0 libsm6 libxrender1 libxext6 \
        libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libgomp1 \
        xvfb ffmpeg screen \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniconda --------------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda config --set always_yes true \
    # Recent conda builds refuse to use the default Anaconda channels until their
    # Terms of Service are accepted; do it non-interactively so the build proceeds.
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda clean -afy

# Code lives OUTSIDE /workspace on purpose: RunPod mounts its persistent volume
# at /workspace by default, which would otherwise shadow the baked-in code.
WORKDIR /opt/pixie

# ---- Conda environment ------------------------------------------------------
# Copy only environment.yaml first so this expensive layer is cached and only
# rebuilds when the env spec changes. Land it in /tmp (NOT /opt/pixie) so the
# WORKDIR stays empty for the `git clone` below — cloning into a non-empty dir
# fails.
COPY environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml && conda clean -afy

# From here on, every RUN executes inside the activated `pixie` env.
SHELL ["conda", "run", "--no-capture-output", "-n", "pixie", "/bin/bash", "-c"]

# ---- MANUAL STEP 2: PyTorch matching CUDA 12.1 ------------------------------
# Must precede every CUDA-compiled extension so they build against this torch.
# Env creation transitively pulls the LATEST torch (violates nerfstudio's
# torch<2.2); this line pins it back to the known-good 2.1.2/cu121.
RUN pip install ninja \
    && pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

# ---- MANUAL STEP 3: tiny-cuda-nn (CUDA-compiled; f3rm feature field) --------
# --no-build-isolation is required: tiny-cuda-nn's setup.py imports both `torch`
# and `pkg_resources` at build time. An isolated build env has neither (it lacks
# torch, and the latest setuptools it pulls in dropped pkg_resources), so build
# against the pixie env, which already has torch and a setuptools with
# pkg_resources. Ensure the build backends are present first.
# TCNN_CUDA_ARCHITECTURES: the build pod usually has no visible GPU, so
# tiny-cuda-nn can't auto-detect the compute capability. It uses its own env var
# (not TORCH_CUDA_ARCH_LIST) and wants the arch numbers WITHOUT dots (86, not
# 8.6), so derive it from TORCH_CUDA_ARCH_LIST by stripping the dots.
RUN export TCNN_CUDA_ARCHITECTURES="$(echo "$TORCH_CUDA_ARCH_LIST" | tr -d '.')" \
    && pip install "setuptools<81" wheel \
    && pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# ---- MANUAL STEP 5 (git parts): PyTorch3D + pinned viewer/CLI deps ----------
# --no-build-isolation: PyTorch3D imports torch in its setup.py, which an
# isolated build env lacks — build it against the pixie env's torch instead.
# tyro>=0.9.8 matches vendored nerfstudio's requirement (an older pin is silently
# upgraded by its install anyway). tyro 0.9.x and jaxtyping need typeguard>=4;
# a stale torchtyping keeps dragging in typeguard 2.x, so pin it forward here.
RUN pip install -v --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    && pip install viser==0.2.7 "tyro>=0.9.8" "typeguard>=4.0.0,<5"

# ---- MANUAL STEP 7: FlashAttention (for Qwen2.5-VL) -------------------------
# Install the official prebuilt wheel instead of compiling from source. flash-attn
# ignores TORCH_CUDA_ARCH_LIST and always builds its heavy sm_80+sm_90 backward
# kernels, which routinely OOM-kills the compiler on a 64 GB box (and takes many
# minutes even when it succeeds). The wheel must match the exact stack:
#   cu122   -> runs on our cu121 runtime (CUDA is compatible within 12.x)
#   torch2.1 / cp310 -> our torch 2.1.2 on Python 3.10
#   cxx11abiFALSE    -> pip torch wheels use the pre-cxx11 ABI (-D_GLIBCXX_USE_CXX11_ABI=0)
# The prebuilt sm_80/sm_90 binaries run on the A6000 (sm_86) just like a source build.
ARG FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
RUN pip install "${FLASH_ATTN_WHEEL}"

# ---- Clone the repo (vendored third_party packages come with it) ------------
# Cloned instead of COPYd so the container is a LIVE git repo: commit + push
# from your host, then `git pull` inside the container to sync. Note this uses
# whatever is PUSHED to ${PIXIE_REF} — local uncommitted changes are NOT baked
# in (push them first).
#
# The repo is private, so the build authenticates with a GitHub token passed as
# a BuildKit secret (never written to a layer). After cloning, the stored remote
# is reset to a token-less URL — the token never lands in the image. The token
# for runtime `git pull` is supplied separately via GIT_ASKPASS (see below).
#
# Build with (build_and_push.sh does this for you):
#   DOCKER_BUILDKIT=1 GITHUB_TOKEN=ghp_xxx \
#     docker build --secret id=github_token,env=GITHUB_TOKEN ...
#
# Docker caches this layer, so a rebuild will NOT re-clone a newer commit unless
# the cache is busted: bump --build-arg PIXIE_REF=... or pass --no-cache. Day to
# day you don't rebuild — you `git pull` inside the running container instead.
ARG PIXIE_REPO=github.com/zaku2dev/pixie.git
ARG PIXIE_REF=main
RUN --mount=type=secret,id=github_token \
    token="$(cat /run/secrets/github_token)" \
    && git clone --branch "${PIXIE_REF}" \
         "https://x-access-token:${token}@${PIXIE_REPO}" /opt/pixie \
    && git -C /opt/pixie remote set-url origin "https://x-access-token@${PIXIE_REPO}" \
    && git config --global --add safe.directory /opt/pixie

# ---- MANUAL STEP 1 + 4 + 6: editable local installs -------------------------
# `-e . --no-deps`: environment.yaml already installed pixie's deps; --no-deps
# avoids re-resolving them (and dodges the arm64-only warp pin issue).
RUN pip install -e . --no-deps \
    && pip install -e third_party/nerfstudio \
    && pip install -e third_party/f3rm \
    && pip install -e third_party/vlmx \
    && pip install -v --no-build-isolation -e third_party/PhysGaussian/gaussian-splatting/submodules/simple-knn/ \
    && pip install -v --no-build-isolation -e third_party/PhysGaussian/gaussian-splatting/submodules/diff-gaussian-rasterization/

# ---- MANUAL STEP 9: known-good pins (guard against transitive drift) --------
RUN pip install --force-reinstall --no-deps numpy==1.24.4 warp_lang==0.10.1

# ---- MANUAL STEP 8: Blender 4.3.2 + add-ons ---------------------------------
# Blender is a standalone binary, not a Python package. Add-ons are downloaded
# as zips; the BLENDER_*_ADDON_PATH env vars set below point config/paths/
# default.yaml at them automatically via its ${oc.env:...} defaults — no manual
# config edit needed.
RUN wget -q "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" -O /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && mkdir -p /opt/blender_addons \
    && wget -q "https://github.com/maximeraafat/BlenderNeRF/archive/refs/tags/${BLENDER_NERF_REF}.zip" \
        -O /opt/blender_addons/BlenderNeRF.zip \
    && wget -q "https://github.com/ReshotAI/gaussian-splatting-blender-addon/archive/${BLENDER_GS_ADDON_REF}.zip" \
        -O /opt/blender_addons/gaussian-splatting-blender-addon.zip \
    && /opt/blender/${BLENDER_SERIES}/python/bin/python3.11 -m ensurepip \
    && /opt/blender/${BLENDER_SERIES}/python/bin/python3.11 -m pip install --upgrade pip objaverse

ENV BLENDER_NERF_ADDON_PATH=/opt/blender_addons/BlenderNeRF.zip \
    BLENDER_GS_ADDON_PATH=/opt/blender_addons/gaussian-splatting-blender-addon.zip

# ---- Runtime ----------------------------------------------------------------
# Reset SHELL and use an entrypoint that activates the conda env for any command
# (interactive shell, `python pipeline.py ...`, etc.).
SHELL ["/bin/bash", "-c"]
# Source conda's shell hook BEFORE activating so interactive shells that only run
# ~/.bashrc (e.g. Vast.ai SSH, which bypasses the ENTRYPOINT) get a working
# `conda activate`. Without the source line, `conda activate` errors with
# "Run 'conda init' before 'conda activate'".
# Also re-export the Blender add-on paths (and GIT_ASKPASS, so `git pull` finds
# the token helper) here: Vast.ai's SSH login shell does NOT inherit the image's
# Docker ENV, so without this the config's ${oc.env:BLENDER_*_ADDON_PATH,...}
# falls back to its non-container default and Blender can't find the add-on zips,
# and `git pull` can't authenticate.
RUN printf '%s\n' \
        'source /opt/conda/etc/profile.d/conda.sh' \
        'conda activate pixie' \
        "export BLENDER_NERF_ADDON_PATH=${BLENDER_NERF_ADDON_PATH}" \
        "export BLENDER_GS_ADDON_PATH=${BLENDER_GS_ADDON_PATH}" \
        'export GIT_ASKPASS=/usr/local/bin/git-askpass.sh' \
        >> /root/.bashrc
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# GIT_ASKPASS lets `git pull` inside the container authenticate to the private
# repo using a GITHUB_TOKEN env var injected at launch, without ever storing the
# token on disk. The remote URL already carries the x-access-token username.
COPY docker/git-askpass.sh /usr/local/bin/git-askpass.sh
ENV GIT_ASKPASS=/usr/local/bin/git-askpass.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/git-askpass.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
