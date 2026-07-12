# =============================================================================
# Pixie — GPU Docker image (conda-based)
# =============================================================================
# Reproduces the full README.md + environment.yaml setup so a fresh GPU instance
# (e.g. RunPod) can run Pixie without redoing the install by hand.
#
# BUILD (run this ON a GPU pod so the long CUDA compiles have a fast machine):
#     docker build \
#       --build-arg TORCH_CUDA_ARCH_LIST="8.6" \   # optional: match your GPU
#       -t <dockerhub-user>/pixie:latest .
#
# The base -devel image ships nvcc, which is required to compile
# tiny-cuda-nn, flash-attn, diff-gaussian-rasterization and PyTorch3D.
# See DOCKER.md for the full RunPod build/push/run walkthrough.
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---- Build-time knobs -------------------------------------------------------
# GPU compute capabilities to compile CUDA extensions for. `docker build` on
# RunPod usually cannot see a GPU, so auto-detection fails — pin the arch(es)
# explicitly. Defaults cover the common RunPod cards:
#   8.0 A100 · 8.6 A6000/A40/A10/3090 · 8.9 4090/L40 · 9.0 H100
# flash-attn requires sm80+, so keep the floor at 8.0. Override to a single
# arch (e.g. "8.6") for a faster build if you only target one card type.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Parallel compile jobs for flash-attn (lower this if the build OOMs the pod).
ARG MAX_JOBS=16
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
        xvfb ffmpeg \
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
# rebuilds when the env spec changes.
COPY environment.yaml ./
RUN conda env create -f environment.yaml && conda clean -afy

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
RUN pip install "setuptools<81" wheel \
    && pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# ---- MANUAL STEP 5 (git parts): PyTorch3D + pinned viewer/CLI deps ----------
RUN pip install -v "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    && pip install viser==0.2.7 tyro==0.6.6

# ---- MANUAL STEP 7: FlashAttention (slow CUDA build; for Qwen2.5-VL) --------
RUN pip install -v -U flash-attn --no-build-isolation

# ---- Copy the repo (vendored third_party packages come with it) -------------
COPY . .

# ---- MANUAL STEP 1 + 4 + 6: editable local installs -------------------------
# `-e . --no-deps`: environment.yaml already installed pixie's deps; --no-deps
# avoids re-resolving them (and dodges the arm64-only warp pin issue).
RUN pip install -e . --no-deps \
    && pip install -e third_party/nerfstudio \
    && pip install -e third_party/f3rm \
    && pip install -e third_party/vlmx \
    && pip install -v -e third_party/PhysGaussian/gaussian-splatting/submodules/simple-knn/ \
    && pip install -v -e third_party/PhysGaussian/gaussian-splatting/submodules/diff-gaussian-rasterization/

# ---- MANUAL STEP 9: known-good pins (guard against transitive drift) --------
RUN pip install --force-reinstall --no-deps numpy==1.24.4 warp_lang==0.10.1

# ---- MANUAL STEP 8: Blender 4.3.2 + add-ons ---------------------------------
# Blender is a standalone binary, not a Python package. Add-ons are downloaded
# as zips; set paths.blender_nerf_addon_path / paths.blender_gs_addon_path in
# config/paths/default.yaml to point at them (see ENV_BLENDER_* below).
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
RUN echo "conda activate pixie" >> /root/.bashrc
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
