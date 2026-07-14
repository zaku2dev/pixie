---
name: pixie-docker-build
description: Build, push, run, and debug the Pixie GPU Docker image. Use when working on the Dockerfile, docker/build_and_push.sh, docker/entrypoint.sh, docker/build_on_ec2.md, DOCKER_RUNPOD.md, DOCKER_VASTAI.md, or when hitting errors during the Docker build (tiny-cuda-nn / flash-attn / PyTorch3D compiles, conda env, CUDA arch), image push, or when running the container on RunPod / Vast.ai / EC2 (BlenderNeRF addon, HuggingFace downloads, git clone/pull auth, Qwen VLM, Blender rendering).
---

# Pixie Docker Build & Debug

Packaged knowledge for building the Pixie GPU image and every failure mode hit so
far. **Read the relevant "Known error" entry before re-diagnosing a build/run
failure** — most have already been solved and the fix is baked into the repo.

## Architecture at a glance

- Base: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` (needs `-devel` for `nvcc`).
- Conda env `pixie` from `environment.yaml`, then a strict sequence of pip/CUDA
  compiles (order matters — see *Ordering constraints*).
- Code is **git-cloned inside the image** (not `COPY`d) so the container is a
  live repo you can `git pull`. Clone is private → needs a token at build time.
- Blender 4.3.2 + BlenderNeRF / gaussian-splatting add-ons are downloaded as
  zips; paths exposed via `BLENDER_*_ADDON_PATH` env → consumed by
  `config/paths/default.yaml` `${oc.env:...}` defaults.
- **Build host must be native x86-64 with lots of RAM; no GPU needed** (arch is
  pinned via build args). Build on EC2 (`docker/build_on_ec2.md`), not Apple Silicon.

## Build & push (the normal path)

```bash
export GITHUB_TOKEN=ghp_xxx                     # read-only PAT for the private repo
IMAGE_REPO=<user>/pixie ./docker/build_and_push.sh 4090   # or a6000
#   PIXIE_REF=main ... to clone a different branch (default: dockerize)
#   MAX_JOBS=4 ...     lower on a 32 GB box to avoid flash-attn OOM
#   ARCH="8.0;8.6;8.9;9.0" ... one image for many cards
```

`build_and_push.sh` requires `GITHUB_TOKEN`, sets `DOCKER_BUILDKIT=1`, and passes
the token as `--secret id=github_token,env=GITHUB_TOKEN` (never baked into a layer).

## GPU / CUDA arch reference

| Preset | Cards | `TORCH_CUDA_ARCH_LIST` | Tag |
|---|---|---|---|
| `4090` | RTX 4090, L40 | `8.9` | `:sm89-4090` |
| `a6000` | A6000, A40, A5000, 3090 | `8.6` | `:sm86-a6000` |
| (manual) | A100 `8.0` · H100 `9.0` | — | — |

- **`tiny-cuda-nn` uses `TCNN_CUDA_ARCHITECTURES` (no dots: `86`, not `8.6`)** —
  derived in the Dockerfile by stripping dots from `TORCH_CUDA_ARCH_LIST`.
- **`flash-attn` ignores the arch list** and always builds sm_80+sm_90 kernels;
  those run fine on sm_86/sm_89.
- Build box usually has no visible GPU → arch **must be pinned**, auto-detect fails.

## Runtime (RunPod / Vast.ai)

```bash
# launch env: -e HF_HOME=/workspace/hf_cache  (weights on persistent disk)
#             -e GITHUB_TOKEN=ghp_xxx         (only if you'll git pull)
#             -e HF_TOKEN=hf_xxx              (avoids HuggingFace 429)
#             -e *_API_KEY=...               (skip for neural MVP / local Qwen)
cd /opt/pixie && git pull                     # sync code without rebuilding
python pipeline.py obj_id=<id> material_mode=vlm paths.base_path=/workspace
```

---

## Known errors & fixes

### 1. BlenderNeRF addon fails to enable
- **Symptom:** `addon_enable` raises / addon not found; previously hardcoded
  `module='BlenderNeRF-main-custom'`.
- **Cause:** Blender enables an addon by its **module name = the top-level folder
  inside the installed zip**. A GitHub archive of tag `v6` unzips to
  `BlenderNeRF-6` (GitHub strips the leading `v`), so the hardcoded name was wrong
  for the `v6` build.
- **Fix (in repo):** `pixie/blender/generate_blendernerf_data.py` derives the name
  from the zip's single top-level folder (`_addon_module_from_zip()`), so it works
  for any zip name/tag. Do **not** re-hardcode it.

### 2. HuggingFace `429 Too Many Requests` on model download
- **Symptom:** `HfHubHTTPError: 429 ... We had to rate limit your IP` from
  `scripts/download_models.py` / `snapshot_download`.
- **Cause:** anonymous downloads are aggressively IP-rate-limited.
- **Fix:** set `HF_TOKEN` (free read token is enough — repo is public, you just
  need to be authenticated). The script now reads `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
  and passes it to `list_repo_files` and `snapshot_download`; also accepts `--token`.
  The 429 is IP-scoped and can take ~5-10 min to clear even after authenticating.

### 3. Build: `git clone ... destination path already exists and is not empty`
- **Cause:** `environment.yaml` used to be `COPY`d into `/opt/pixie` before the
  clone, leaving the WORKDIR non-empty.
- **Fix (in repo):** `environment.yaml` is copied to `/tmp/environment.yaml`
  instead, keeping `/opt/pixie` empty for the clone. Preserve this — don't move
  the copy back into the WORKDIR.

### 4. Build clone auth fails / `Set GITHUB_TOKEN`
- **Cause:** repo is private; the in-image `git clone` needs credentials.
- **Fix:** export `GITHUB_TOKEN` before building; `build_and_push.sh` forwards it
  as a BuildKit secret. The clone embeds the token in the URL, then **resets the
  remote to a token-less `https://x-access-token@github.com/...` URL** so the token
  never persists in the image. Never pass the token via `--build-arg` (it lands in
  layer history) or EC2 user-data (readable via IMDS). On EC2 use
  `read -rsp 'GitHub token: ' GITHUB_TOKEN; export GITHUB_TOKEN`.

### 5. Runtime `git pull` auth fails
- **Cause:** the container remote is token-less by design.
- **Fix:** `GIT_ASKPASS=/usr/local/bin/git-askpass.sh` feeds `$GITHUB_TOKEN` to git
  at pull time without writing it to disk; the remote URL carries the
  `x-access-token` username. Just launch with `-e GITHUB_TOKEN=...` and `git pull`.

### 6. Vast.ai SSH: Blender addon not found / `git pull` unauth / `conda activate` errors
- **Cause:** Vast.ai's SSH login shell **bypasses the image `ENTRYPOINT` and does
  NOT inherit Docker `ENV`**. So `BLENDER_*_ADDON_PATH`, `GIT_ASKPASS`, and the
  conda hook are all missing in that shell.
- **Fix (in repo):** the Dockerfile writes `source conda.sh`, `conda activate
  pixie`, and re-exports `BLENDER_NERF_ADDON_PATH`, `BLENDER_GS_ADDON_PATH`,
  `GIT_ASKPASS` into `/root/.bashrc`. Any new runtime env var a Vast SSH shell
  needs must also be added here, not just as `ENV`.

### 7. flash-attn OOM-kills the compiler
- **Symptom:** build dies during flash-attn; host runs out of RAM.
- **Cause:** flash-attn compiles heavy sm_80+sm_90 backward kernels; each nvcc job
  uses ~3-6 GB, so peak RAM ≈ `MAX_JOBS × 6 GB`.
- **Fix (in repo):** install the **prebuilt wheel** (`FLASH_ATTN_WHEEL`, matched to
  cu122/torch2.1/cp310/cxx11abiFALSE) instead of compiling. If you ever compile
  from source, lower `MAX_JOBS` (4 on a 32 GB box).

### 8. torch version drift breaks CUDA extensions
- **Cause:** `conda env create` transitively pulls the **latest** torch (violates
  nerfstudio's `torch<2.2`); CUDA extensions must build against the runtime torch.
- **Fix (in repo):** pin `torch==2.1.2 torchvision==0.16.2 --index-url .../cu121`
  **before** any CUDA-compiled extension (tiny-cuda-nn, flash-attn, PyTorch3D,
  gaussian rasterizer). Never reorder these after the compile steps.

### 9. tiny-cuda-nn / PyTorch3D build-isolation failures
- **Cause:** their `setup.py` imports `torch` (and tcnn also `pkg_resources`) at
  build time; an isolated build env lacks both (latest setuptools dropped
  `pkg_resources`).
- **Fix (in repo):** `pip install --no-build-isolation` against the pixie env, with
  `setuptools<81 wheel` installed first for `pkg_resources`.

### 10. conda refuses default channels (ToS)
- **Symptom:** env creation aborts asking to accept channel Terms of Service.
- **Fix (in repo):** `conda tos accept --override-channels --channel .../pkgs/main`
  (and `/pkgs/r`) run non-interactively before `conda env create`.

### 11. Silent numpy / warp version drift
- **Fix (in repo):** `pip install --force-reinstall --no-deps numpy==1.24.4
  warp_lang==0.10.1` as a late guard against transitive upgrades. Keep it last.

### 12. Build fails/crashes on Apple Silicon
- **Cause:** the CUDA/conda x86 binaries fail under Rosetta emulation.
- **Fix:** build on a **native x86-64 Linux host** (EC2 `m7i.4xlarge`; `c7i.4xlarge`
  + `MAX_JOBS=4`). See `docker/build_on_ec2.md`.

### 13. Rebuild doesn't pick up new commits
- **Cause:** Docker caches the `git clone` layer.
- **Fix:** bust it with `--build-arg PIXIE_REF=<ref>` (changing the ref) or
  `--no-cache`. Day-to-day, prefer `git pull` inside the running container.

### 14. `.dockerignore` breaks a `COPY`
- **Note:** `docker/` must **not** be ignored — the Dockerfile `COPY`s
  `docker/entrypoint.sh` and `docker/git-askpass.sh`; an ignored path can't be
  `COPY`d ("not found"). Large data/model dirs are intentionally ignored.

### 15. Qwen VLM issues (local, no-API path)
- **Model id `o` collision:** the dispatcher (`third_party/vlmx/vlmx/prompt_utils.py`)
  checks `'o' in model_name` (GPT branch) **before** the qwen branch — never use a
  custom id containing `o`. Standard `Qwen/Qwen2.5-VL-*` ids are safe.
- **VRAM:** 7B (bf16 ≈16 GB) is tight-but-usable on a 4090 (24 GB), comfortable on
  A6000 (48 GB); 32B/72B need A6000 or multi-GPU. `QwenWrapper` loads with
  `device_map="auto"`, so on a **2×4090** box it shards across both GPUs (48 GB
  aggregate) — but the neural stages stay single-GPU.
- Default config uses **Gemini**, not Qwen; Qwen only loads if you set a Qwen id in
  `config/segmentation/default.yaml`.

---

## Ordering constraints (do not reorder)

1. apt + git-lfs → 2. conda env (`/tmp/environment.yaml`) → 3. **pin torch
   2.1.2/cu121** → 4. tiny-cuda-nn (`--no-build-isolation`, `TCNN_CUDA_ARCHITECTURES`)
   → 5. PyTorch3D + viser/tyro → 6. flash-attn prebuilt wheel → 7. **git clone repo
   into empty `/opt/pixie`** → 8. editable installs (`pip install -e .` + `third_party/*`)
   → 9. numpy/warp pins → 10. Blender + add-ons → 11. `.bashrc` re-exports + entrypoint.

Torch must precede every CUDA compile; the clone must precede the editable
installs; the numpy/warp pin must come after everything that could upgrade them.

## Debugging tips

- **Reproduce a failing build stage fast:** the arch (`TORCH_CUDA_ARCH_LIST`) is an
  `ENV` near the top, so changing it invalidates the cache for *every* later layer.
  Iterate with a fixed arch to keep the cache warm.
- **Inspect the built image without a GPU:** `docker run --rm -it <image> bash` —
  the entrypoint activates the `pixie` env. GPU code needs `--gpus all` on a GPU host.
- **Confirm addon module name:** `unzip -l <addon>.zip | head` shows the top-level
  folder Blender will enable.
- **HF/token issues:** unset means anonymous → rate limited; a warning is printed by
  `download_models.py` when no token is found.
