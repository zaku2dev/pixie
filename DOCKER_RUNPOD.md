# Deploying Pixie on RunPod (step by step)

This bakes the entire README.md install (conda env + all CUDA-compiled
extensions + Blender 4.3.2) into a single Docker image, so a fresh GPU instance
runs Pixie without redoing the setup by hand. Build the image **once**, push it
to a registry, and every future RunPod pod pulls the finished image in seconds.

**What is / isn't in the image**

| In the image | Not in the image (download / mount at runtime) |
|---|---|
| conda `pixie` env (`environment.yaml`) | Model checkpoints (`scripts/download_models.py`) |
| torch 2.1.2 / cu121 | PixieVerse dataset (`scripts/download_data.py`) |
| tiny-cuda-nn, flash-attn, PyTorch3D | API keys (OpenAI/Claude/Gemini) |
| nerfstudio, f3rm, vlmx, PhysGaussian submodules | Your generated outputs |
| Blender 4.3.2 + BlenderNeRF/GS add-ons | |

Code lives at **`/opt/pixie`** inside the image. Large, code-independent models
and data live on a **persistent volume mounted at `/workspace`**, and
`paths.base_path=/workspace` points Pixie at them. (Code is deliberately kept
out of `/workspace` so RunPod's volume doesn't shadow it.)

> **Deploying on Vast.ai instead?** The image is platform-agnostic — Step 1
> (build) is the same everywhere. Vast.ai is handy when A6000s are scarce on
> RunPod; see **[DOCKER_VASTAI.md](DOCKER_VASTAI.md)** for that walkthrough.

---

## Step 0 — Prerequisites

- A **Docker Hub** account (or any registry) to host the image — note your
  username; the image will be `<user>/pixie:<tag>`.
- A **RunPod** account with credits.
- (Optional) VLM **API keys** (OpenAI / Anthropic / Gemini) — only needed for
  `material_mode=vlm` labeling. The neural physics-estimation MVP needs none.

---

## Step 1 — Build and push the image

`docker build` compiles tiny-cuda-nn, PyTorch3D and the gaussian rasterizer from
source (flash-attn installs as a prebuilt wheel) — this needs a machine with a
**Docker daemon**, but **not a GPU** (the CUDA arch is baked in via a build arg,
not detected).

> ⚠️ **Must be a native x86-64 (amd64) Linux host.** RunPod GPUs are x86-64, so
> the image targets `linux/amd64` (the script passes `--platform linux/amd64`).
> **Apple Silicon Macs cannot build it** — the amd64 CUDA/conda binaries crash
> under Rosetta emulation (`rosetta error: ... ld-linux-x86-64.so.2`, exit 133),
> and even if they didn't, the source compiles would take hours under emulation.
> Intel Macs / Linux-x86_64 / WSL2-on-x86 are fine.

Two ways to get a suitable host:

**Option A — an x86-64 Linux box with Docker (recommended, simplest).**
An x86-64 cloud VM (even CPU-only; ~16 vCPU / 64 GB RAM builds comfortably), an
Intel Linux workstation, or WSL2 on an x86 PC. No GPU required.
👉 For a copy-paste AWS walkthrough, see
[docker/build_on_ec2.md](docker/build_on_ec2.md).

**Option B — a RunPod pod.**
Standard RunPod pods run *inside* a container with **no Docker daemon**, so
`docker build` fails there. You need a host that exposes Docker — a pod/template
built for docker-in-docker, or a bare cloud VM with Docker + the NVIDIA
Container Toolkit. If that friction isn't worth it, use Option A: the build
never touches the GPU anyway.

On whichever host:

The code is **git-cloned inside the image** (not copied), so the build needs a
**GitHub token** with read access to the repo, exported as `GITHUB_TOKEN`. It is
forwarded as a BuildKit secret and never baked into a layer. The build clones
whatever is **pushed** to the branch (`PIXIE_REF`, default `main`) — push
first. At runtime the container is a live repo you can `git pull` (set
`-e GITHUB_TOKEN=...` at launch; the token is fed to git via `GIT_ASKPASS`, never
stored on disk).

```bash
git clone <your-fork-url> pixie && cd pixie
docker login                       # authenticate to Docker Hub once
export GITHUB_TOKEN=ghp_xxxxxxxx   # PAT with read access to the pixie repo

# Pick the preset matching the GPU you will RUN on:
# RTX 4090 / L40 (sm_89) — best value for the neural pipeline:
IMAGE_REPO=<dockerhub-user>/pixie ./docker/build_and_push.sh 4090
#   -> pushes <dockerhub-user>/pixie:sm89-4090

# A6000 / A40 / A5000 / 3090 (sm_86) — 48 GB, e.g. local Qwen VLM:
IMAGE_REPO=<dockerhub-user>/pixie ./docker/build_and_push.sh a6000
#   -> pushes <dockerhub-user>/pixie:sm86-a6000
```

| Preset | GPUs | `TORCH_CUDA_ARCH_LIST` | Image tag |
|---|---|---|---|
| `4090`  | RTX 4090, L40           | `8.9` | `:sm89-4090` |
| `a6000` | A6000, A40, A5000, 3090 | `8.6` | `:sm86-a6000` |

Each preset builds a single CUDA arch (fastest build, smallest image) and tags a
distinct image, so both can live in your registry. Need another card
(A100 `8.0`, H100 `9.0`) or one image that runs on all of them? Override the
arch:

```bash
IMAGE=<dockerhub-user>/pixie:multi ARCH="8.0;8.6;8.9;9.0" \
  ./docker/build_and_push.sh a6000   # preset arg required; ARCH wins
```

A multi-arch build recompiles the source CUDA extensions (tiny-cuda-nn,
PyTorch3D, the gaussian rasterizer) for every listed arch, so it's much slower.
On a bigger box, raise the compile parallelism to claw some of that back, e.g.
prefix with `MAX_JOBS=16`.

The build takes a while (tiny-cuda-nn/PyTorch3D and the gaussian rasterizer
compile from source; flash-attn installs as a prebuilt wheel). When it finishes,
confirm the tag is visible in your Docker Hub repo.

> **Private image?** If your Docker Hub repo is private, you'll add registry
> credentials to the RunPod template in Step 3. Making it public is simpler for
> a first deploy.

---

## Step 2 — Create a persistent volume on RunPod

So models/data/outputs survive pod restarts and are reusable across pods:

1. RunPod console → **Storage** → **＋ Network Volume**.
2. Pick a **region**, give it a name (e.g. `pixie-store`), and a size
   (**≥ 100 GB** — checkpoints + a dataset class + render outputs add up).
3. Create it. Note the region — your pod must be deployed in the **same region**
   to attach this volume.

(Skip this only for a throwaway test; then use the pod's ephemeral volume disk
instead and expect to re-download models each time.)

---

## Step 3 — Create a RunPod template

A template pins the image + mount path + env vars so you can launch identical
pods repeatedly:

1. RunPod console → **Templates** → **New Template**.
2. **Container Image:** `<dockerhub-user>/pixie:sm89-4090`
   (or your `sm86-a6000` tag — match the GPU you'll select in Step 4).
3. **Container Disk:** ~30–40 GB (the image itself is large).
4. **Volume Mount Path:** `/workspace`  ← must stay `/workspace` so it matches
   `paths.base_path=/workspace` and does **not** shadow the code at `/opt/pixie`.
5. **Environment Variables:**
   - VLM labeling via cloud API (skip for the neural MVP or the local-Qwen
     path): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.
   - `HF_HOME=/workspace/hf_cache` — recommended so Hugging Face model weights
     (Qwen VLM, and download caches) land on the persistent volume instead of
     the ephemeral container disk.
6. **Registry Credentials:** add your Docker Hub login **only if** the image
   repo is private.
7. Leave the default `ENTRYPOINT`/`CMD` — the image already activates the conda
   `pixie` env for you. Save.

---

## Step 4 — Deploy a pod

1. RunPod console → **Pods** → **＋ Deploy** (in the **same region** as your
   volume from Step 2).
2. **GPU:** choose **RTX 4090** for the `sm89-4090` image (or A6000/A40 for
   `sm86-a6000`). 24 GB is plenty for single-object neural inference.
3. **Template:** select the template from Step 3.
4. **Network Volume:** attach `pixie-store` (mounts at `/workspace`).
5. Deploy. When it's **Running**, open a shell: pod → **Connect** →
   *Start Web Terminal* (or SSH). You land in `/opt/pixie` with the `pixie`
   conda env already active.

---

## Step 5 — First run inside the pod

`paths.base_path=/workspace` makes Pixie read/write `data/`, `models/`,
`render_outputs/`, `mpm_sim_outputs/`, checkpoints, etc. on the persistent
volume. Download the models (and, if training/eval, a dataset class) there once:

```bash
# You are in /opt/pixie with the `pixie` env active.

# 1. Model checkpoints -> persistent volume:
python scripts/download_models.py --local-dir /workspace

# 2. (Optional) one dataset class for quick testing:
python scripts/download_data.py --dataset-repo vlongle/pixieverse \
    --dirs archives --obj-class tree --local-dir /workspace

# 3. MVP: neural physical-parameter estimation (no VLM/API key needed):
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    material_mode=neural paths.base_path=/workspace
```

> Tip: to avoid repeating `paths.base_path=/workspace`, set `base_path:
> /workspace` once in `config/paths/default.yaml` (edit it on the volume, or
> bake it into your fork before building).

**Blender rendering** (`render.py` shells out to `blender`, already on `PATH`).
Headless pods have no display, so wrap it in `xvfb`:

```bash
xvfb-run -a python render.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    paths.base_path=/workspace \
    paths.blender_nerf_addon_path=$BLENDER_NERF_ADDON_PATH \
    paths.blender_gs_addon_path=$BLENDER_GS_ADDON_PATH
```

### Blender add-on paths

The add-on zips ship in the image at pinned versions (BlenderNeRF tag `v6`; the
GS add-on at commit `dad6545`, since neither ships a release asset):
- `/opt/blender_addons/BlenderNeRF.zip`  → `paths.blender_nerf_addon_path`
- `/opt/blender_addons/gaussian-splatting-blender-addon.zip` → `paths.blender_gs_addon_path`

These are exported as `$BLENDER_NERF_ADDON_PATH` / `$BLENDER_GS_ADDON_PATH`, and
`config/paths/default.yaml` already reads those env vars by default — so the
paths resolve automatically. Override on the CLI (as above) only if you use a
different zip.

> **Custom fork caveat:** `config/paths/default.yaml` defaults to
> `BlenderNeRF-main-custom.zip`, implying the paper used a *customized*
> BlenderNeRF. If so, rebuild with `--build-arg BLENDER_NERF_REF=...` or
> bind-mount your own zip and repoint `paths.blender_nerf_addon_path` at it.

### Fully local (no API keys) with Qwen

The VLM labeling pipeline (`material_mode=vlm`) defaults to cloud models
(Gemini/Claude/GPT). You can run it **entirely on the pod's GPU with no API
keys** by switching the model ids to a local Qwen2.5-VL checkpoint. The
dispatcher (`vlmx/prompt_utils.py`) routes any model name containing `qwen` to a
local `Qwen2_5_VLForConditionalGeneration` load — flash-attn (already in the
image) is required and used automatically.

1. Point the labeling stages at a Qwen HF repo id in
   `config/segmentation/default.yaml` (the string is passed verbatim to
   `from_pretrained`, so use a real id like `Qwen/Qwen2.5-VL-7B-Instruct`):
   ```yaml
   vlm:
     labeling:
       models:
         data_filtering: "Qwen/Qwen2.5-VL-7B-Instruct"
         segmentation:   "Qwen/Qwen2.5-VL-7B-Instruct"
         seg_critic:     "Qwen/Qwen2.5-VL-7B-Instruct"
         phys_sampler:   "Qwen/Qwen2.5-VL-7B-Instruct"
         phys_judge:     "Qwen/Qwen2.5-VL-7B-Instruct"
         parse_critic:   "Qwen/Qwen2.5-VL-7B-Instruct"
   ```
2. Ensure `HF_HOME=/workspace/hf_cache` (Step 3) so the multi-GB weights
   download **once** to the persistent volume and are reused on later pods.
3. No `*_API_KEY` env vars are needed — you can remove them from the template.

```bash
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    material_mode=vlm paths.base_path=/workspace
```

**GPU sizing for local Qwen:** the 7B checkpoint (bf16) fits comfortably on an
**A6000 (48 GB)** and is tight-but-usable on a **4090 (24 GB)** given the
long multi-view prompts; the 32B/72B variants need the A6000 or multi-GPU. If
you plan to run local Qwen, build/deploy the **`a6000`** image. Do **not** use a
custom model id containing the letter `o` — the dispatcher checks the GPT branch
(`'o' in model_name`) before Qwen; standard `Qwen/Qwen2.5-VL-*` ids are safe.

---

## Step 6 — Stop, restart, and persistence

- **Stop** the pod when idle to stop paying for the GPU. Anything on
  `/workspace` (the network volume) persists; anything elsewhere in the
  container (including `/opt/pixie` code edits) is lost on a fresh pod.
- **Restart** later by deploying a new pod from the same template + volume — no
  re-download needed; models are already on the volume.
- **Iterating on code:** for a quick change, edit under `/opt/pixie` in a
  running pod (editable `pip install -e` picks it up immediately) — but that
  edit is ephemeral. For anything you want to keep, commit to your fork and
  rebuild the image (only needed when `environment.yaml` or a compiled extension
  changes).

---

## Notes & tuning

- **Build OOM:** `MAX_JOBS` caps parallel jobs for the source-compiled CUDA
  extensions (flash-attn is a prebuilt wheel, so it's no longer the RAM
  bottleneck). Default `6` fits a 64 GB box; on less RAM lower it —
  `MAX_JOBS=4` on 32 GB, `2` on 16 GB, e.g.
  `MAX_JOBS=4 IMAGE_REPO=... ./docker/build_and_push.sh 4090`; on a bigger box
  raise it (e.g. `MAX_JOBS=16`) for a faster build.
- **CUDA version:** the image targets CUDA 12.1 (torch 2.1.2/cu121). The host
  NVIDIA driver just needs to support CUDA ≥ 12.1; the toolkit is inside the
  image, so RunPod's driver is fine.
- **Wrong-arch image won't launch:** an `sm89-4090` image runs only on Ada
  cards. If you deploy it on an A6000 it will fail — match the pod GPU to the
  image tag, or build the `multi` arch image.
- **Training the UNet** (`third_party/Wavelet-Generation/trainer/...`) is the
  heavy multi-GPU path (paper: 6× A6000). For that, deploy a multi-GPU pod with
  the `sm86-a6000` image and scale `training.training.batch_size` to your card
  count; the physical-parameter-estimation MVP above does **not** need it.
