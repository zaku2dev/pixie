# Pixie in Docker (RunPod GPU workflow)

This bakes the entire README.md install (conda env + all CUDA-compiled
extensions + Blender 4.3.2) into a single image, so a new GPU instance is ready
to run Pixie without redoing the setup by hand.

**What is / isn't in the image**

| In the image | Not in the image (mount / download at runtime) |
|---|---|
| conda `pixie` env (`environment.yaml`) | Model checkpoints (`scripts/download_models.py`) |
| torch 2.1.2 / cu121 | PixieVerse dataset (`scripts/download_data.py`) |
| tiny-cuda-nn, flash-attn, PyTorch3D | API keys (OpenAI/Claude/Gemini) |
| nerfstudio, f3rm, vlmx, PhysGaussian submodules | Your generated outputs |
| Blender 4.3.2 + BlenderNeRF/GS add-ons | |

Models and data are large and change independently of the code, so keep them on
a mounted volume rather than rebuilding the image for every dataset refresh.

---

## 1. Prerequisites

- A Docker Hub (or other registry) account for storing the built image.
- A RunPod account with credits.

## 2. Build the image on a GPU pod

`docker build` compiles tiny-cuda-nn, flash-attn, PyTorch3D and the gaussian
rasterizer from source — slow on a laptop, fast on a GPU pod. Build there once,
push to a registry, and every future pod pulls the finished image in seconds.

1. **Launch a build pod** on RunPod. Pick a template that provides a Docker
   daemon (RunPod's *"RunPod Pytorch"* pods run in containers without one; use a
   host that gives you Docker access — e.g. a RunPod pod based on a
   `docker:dind` template, or any cloud VM with an NVIDIA GPU + Docker + the
   NVIDIA Container Toolkit). Any Ampere-or-newer GPU works.

2. **Clone and build:**
   ```bash
   git clone <your-fork-url> pixie && cd pixie

   # Match ARCH to the GPU you will RUN on (not necessarily the build GPU):
   #   8.0 A100 · 8.6 A6000/A40/A10/3090 · 8.9 4090/L40 · 9.0 H100
   IMAGE=<dockerhub-user>/pixie:latest ARCH=8.6 ./docker/build_and_push.sh
   ```
   The default `ARCH` (`8.0;8.6;8.9;9.0`) builds fat binaries for all common
   RunPod cards — bigger and slower to compile, but runs anywhere. Pin a single
   arch for a faster build if you know your target GPU.

   > `docker build` cannot see the GPU, so the arch list is passed explicitly
   > via `--build-arg` (already handled by the script). Do not rely on
   > auto-detection.

3. `build_and_push.sh` runs `docker login` implicitly via your existing
   credentials — run `docker login` first if you haven't.

## 3. Run on a fresh GPU pod

Create a new RunPod pod (or `docker run`) from the pushed image. All Pixie
inputs/outputs live under `paths.base_path` (which defaults to `null` and MUST
be set), so mount one persistent volume and point `base_path` at it:

```bash
docker run --gpus all -it --rm \
  -v /workspace/pixie_store:/store \
  -e OPENAI_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  -e GEMINI_API_KEY=... \
  <dockerhub-user>/pixie:latest
```

On RunPod, set the image to `<dockerhub-user>/pixie:latest`, attach a volume at
`/store`, and add the API keys as pod environment variables.

The container starts with the conda `pixie` env already activated.

## 4. First-run inside the container

`paths.base_path` derives `data/`, `models/`, `render_outputs/`,
`mpm_sim_outputs/`, checkpoints, etc. Set it once (CLI or edit
`config/paths/default.yaml`) so everything lands on the mounted `/store` volume:

```bash
# Pull model checkpoints and (optionally) the dataset onto the mounted volume:
python scripts/download_models.py
python scripts/download_data.py --dataset-repo vlongle/pixieverse \
    --dirs archives --obj-class tree --local-dir /store

# Neural material prediction (no VLM key needed):
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    material_mode=neural paths.base_path=/store

# Blender rendering (render.py shells out to `blender`, which is on PATH).
# Headless pods have no display, so wrap the whole command in xvfb:
xvfb-run -a python render.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    paths.base_path=/store \
    paths.blender_nerf_addon_path=$BLENDER_NERF_ADDON_PATH \
    paths.blender_gs_addon_path=$BLENDER_GS_ADDON_PATH
```

### Blender add-on paths

The add-on zips ship in the image at pinned versions (BlenderNeRF tag `v6`; the
GS add-on at commit `dad6545`, since neither ships a release asset):
- `/opt/blender_addons/BlenderNeRF.zip`  → `paths.blender_nerf_addon_path`
- `/opt/blender_addons/gaussian-splatting-blender-addon.zip` → `paths.blender_gs_addon_path`

(also exported as `$BLENDER_NERF_ADDON_PATH` / `$BLENDER_GS_ADDON_PATH`.)
Point `config/paths/default.yaml` at these, or override on the CLI (as shown in
the `render.py` example above).

> **Custom fork caveat:** `config/paths/default.yaml` defaults to
> `BlenderNeRF-main-custom.zip`, implying the paper used a *customized*
> BlenderNeRF. If so, override the build arg (`--build-arg BLENDER_NERF_REF=...`)
> or bind-mount your own zip and repoint `paths.blender_nerf_addon_path` at it.

## 5. Notes & tuning

- **Build OOM (flash-attn):** lower parallelism —
  `MAX_JOBS=4 IMAGE=... ./docker/build_and_push.sh`.
- **CUDA version:** the image targets CUDA 12.1 (torch 2.1.2/cu121). The host
  NVIDIA driver just needs to support CUDA ≥ 12.1; the toolkit is inside the
  image.
- **Iterating on code:** mount your working copy over `/workspace/pixie` to edit
  without rebuilding — the editable (`pip install -e`) installs pick up changes.
  Rebuild only when `environment.yaml` or a compiled extension changes.
