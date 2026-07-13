# Deploying Pixie on Vast.ai (step by step)

This bakes the entire README.md install (conda env + all CUDA-compiled
extensions + Blender 4.3.2) into a single Docker image, so a fresh GPU instance
runs Pixie without redoing the setup by hand. Build the image **once**, push it
to a registry, and every future Vast.ai instance pulls the finished image in
seconds.

Vast.ai rents bare instances from a host marketplace, so **A6000 offers are
often available when RunPod is dry** — this guide targets the `sm86-a6000`
image. The image itself is platform-agnostic (the build in Step 1 is identical
to the RunPod flow); only the deploy differs. The one real difference to plan
for: Vast has **no drop-in network volume** — an instance's own disk is its
storage (see Step 4).

> Deploying on RunPod instead? See **[DOCKER_RUNPOD.md](DOCKER_RUNPOD.md)**.

**What is / isn't in the image**

| In the image | Not in the image (download / mount at runtime) |
|---|---|
| conda `pixie` env (`environment.yaml`) | Model checkpoints (`scripts/download_models.py`) |
| torch 2.1.2 / cu121 | PixieVerse dataset (`scripts/download_data.py`) |
| tiny-cuda-nn, flash-attn, PyTorch3D | API keys (OpenAI/Claude/Gemini) |
| nerfstudio, f3rm, vlmx, PhysGaussian submodules | Your generated outputs |
| Blender 4.3.2 + BlenderNeRF/GS add-ons | |

Code lives at **`/opt/pixie`** inside the image. Large, code-independent models
and data live under **`paths.base_path` (default `/workspace`)** — on Vast.ai
that's just a directory on the instance's own disk (there's no separate network
volume by default; see Step 4). Code is deliberately kept out of `/workspace` so
a mounted volume can't shadow it.

---

## Step 0 — Prerequisites

- A **Docker Hub** account (or any registry) to host the image — note your
  username; the image will be `<user>/pixie:<tag>`.
- A **Vast.ai** account with credit. For the scripted path, install the CLI:
  `pip install --upgrade vastai` then `vastai set api-key <your-key>`.
- (Optional) VLM **API keys** (OpenAI / Anthropic / Gemini) — only needed for
  `material_mode=vlm` labeling. The neural physics-estimation MVP needs none.

---

## Step 1 — Build and push the image

`docker build` compiles tiny-cuda-nn, flash-attn, PyTorch3D and the gaussian
rasterizer from source — this needs a machine with a **Docker daemon**, but
**not a GPU** (the CUDA arch is baked in via a build arg, not detected).

> ⚠️ **Must be a native x86-64 (amd64) Linux host.** Cloud GPU hosts (Vast.ai,
> RunPod, …) are x86-64, so the image targets `linux/amd64` (the script passes
> `--platform linux/amd64`). **Apple Silicon Macs cannot build it** — the amd64
> CUDA/conda binaries crash under Rosetta emulation (`rosetta error: ...
> ld-linux-x86-64.so.2`, exit 133), and even if they didn't, the source compiles
> would take hours under emulation. Intel Macs / Linux-x86_64 / WSL2-on-x86 are
> fine.

Easiest host is an **x86-64 Linux box with Docker** — an x86-64 cloud VM (even
CPU-only; ~16 vCPU / 64 GB RAM builds comfortably), an Intel Linux workstation,
or WSL2 on an x86 PC. No GPU required.
👉 For a copy-paste AWS walkthrough, see
[docker/build_on_ec2.md](docker/build_on_ec2.md).

On the build host:

```bash
git clone <your-fork-url> pixie && cd pixie
docker login                       # authenticate to Docker Hub once

# A6000 / A40 / A5000 / 3090 (sm_86) — the target for this guide:
IMAGE_REPO=<dockerhub-user>/pixie ./docker/build_and_push.sh a6000
#   -> pushes <dockerhub-user>/pixie:sm86-a6000

# RTX 4090 / L40 (sm_89) — build this too if you might rent a 4090 offer:
IMAGE_REPO=<dockerhub-user>/pixie ./docker/build_and_push.sh 4090
#   -> pushes <dockerhub-user>/pixie:sm89-4090
```

| Preset | GPUs | `TORCH_CUDA_ARCH_LIST` | Image tag |
|---|---|---|---|
| `a6000` | A6000, A40, A5000, 3090 | `8.6` | `:sm86-a6000` |
| `4090`  | RTX 4090, L40           | `8.9` | `:sm89-4090` |

Each preset builds a single CUDA arch (fastest build, smallest image) and tags a
distinct image, so both can live in your registry. Need another card
(A100 `8.0`, H100 `9.0`) or one image that runs on all of them? Override the
arch:

```bash
IMAGE=<dockerhub-user>/pixie:multi ARCH="8.0;8.6;8.9;9.0" \
  ./docker/build_and_push.sh a6000   # preset arg required; ARCH wins
```

A multi-arch build compiles flash-attn for every listed arch, so it's much
slower. On a box with more than 64 GB RAM, raise the compile parallelism to
claw some of that back, e.g. prefix with `MAX_JOBS=16`.

The build takes a while (flash-attn/tiny-cuda-nn compile from source). When it
finishes, confirm the tag is visible in your Docker Hub repo.

> **Private image?** If your Docker Hub repo is private, add your registry
> credentials when you launch the instance in Step 2. Making it public is
> simpler for a first deploy.

---

## Step 2 — Launch an A6000 instance

Vast.ai has no reusable "template" object like RunPod — you set the image, disk,
and env vars directly when you rent an offer. Two ways:

### Option A — web console
1. Console → **Search** (Create/Rent).
2. **Filter GPU:** *GPU Type = RTX A6000*, and set **Disk Space ≥ 100 GB** (the
   image + checkpoints + a dataset class add up). Sort by $/hr or reliability.
3. **Edit image & config** → set the image to
   `<dockerhub-user>/pixie:sm86-a6000`. Add your Docker Hub credentials here if
   the repo is private.
4. **Launch Mode: SSH** (interactive shell). Vast adds its own SSH server, so it
   bypasses the image's `ENTRYPOINT` — but the image also writes
   `conda activate pixie` into `/root/.bashrc`, so your SSH shell still lands
   with the `pixie` env active. (Pick "Docker ENTRYPOINT" mode instead and the
   image's own entrypoint runs and activates the env too.)
5. **Env / Docker options:** in the *Docker options* field add
   `-e HF_HOME=/workspace/hf_cache` (so Hugging Face weights land on the instance
   disk, not the tiny default cache) plus any `*_API_KEY`s. Skip the keys for the
   neural MVP or the local-Qwen path.
6. **Rent** the offer. When it shows **Running**, copy the SSH command from the
   instance card and connect. You land in a shell with the `pixie` env active.

### Option B — CLI (reproducible)
```bash
# Cheapest single-A6000 offer with enough disk (ID is the first column):
vastai search offers 'gpu_name=RTX_A6000 num_gpus=1 disk_space>=100 rentable=true' -o 'dph+'

vastai create instance <OFFER_ID> \
  --image <dockerhub-user>/pixie:sm86-a6000 \
  --disk 120 \
  --ssh --direct \
  --env '-e HF_HOME=/workspace/hf_cache'    # append -e OPENAI_API_KEY=... etc. as needed

vastai show instances                        # print the SSH host/port once it's running
```
(Flag names vary slightly by CLI version — run `vastai create instance --help`.
For a private image, register your Docker Hub login in your Vast account
settings first.)

---

## Step 3 — First run inside the instance

`paths.base_path=/workspace` makes Pixie read/write `data/`, `models/`,
`render_outputs/`, `mpm_sim_outputs/`, checkpoints, etc. under `/workspace` on
the instance disk. Download the models (and, if training/eval, a dataset class)
there once:

```bash
# You are in a shell with the `pixie` env active.

# 1. Model checkpoints -> instance disk:
python scripts/download_models.py --local-dir /workspace

# 2. (Optional) one dataset class for quick testing:
python scripts/download_data.py --dataset-repo vlongle/pixieverse \
    --dirs archives --obj-class tree --local-dir /workspace

# 3. MVP: neural physical-parameter estimation (no VLM/API key needed):
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    material_mode=neural paths.base_path=/workspace
```

> Tip: to avoid repeating `paths.base_path=/workspace`, set `base_path:
> /workspace` once in `config/paths/default.yaml` (edit it on the instance, or
> bake it into your fork before building).

**Blender rendering** (`render.py` shells out to `blender`, already on `PATH`).
Headless instances have no display, so wrap it in `xvfb`:

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
(Gemini/Claude/GPT). You can run it **entirely on the instance GPU with no API
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
2. Ensure `HF_HOME=/workspace/hf_cache` (set at launch, Step 2) so the multi-GB
   weights download **once** to the instance disk. To reuse them across
   *different* instances, keep them on a Vast.ai Volume (see Step 4).
3. No `*_API_KEY` env vars are needed — you can omit them from the launch config.

```bash
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 \
    material_mode=vlm paths.base_path=/workspace
```

**GPU sizing for local Qwen:** the 7B checkpoint (bf16) fits comfortably on an
**A6000 (48 GB)** and is tight-but-usable on a **4090 (24 GB)** given the
long multi-view prompts; the 32B/72B variants need the A6000 or multi-GPU. The
A6000 is the recommended target here. Do **not** use a custom model id
containing the letter `o` — the dispatcher checks the GPT branch
(`'o' in model_name`) before Qwen; standard `Qwen/Qwen2.5-VL-*` ids are safe.

---

## Step 4 — Storage, stop, and persistence

Vast.ai has **no shared network volume** by default: the instance disk you sized
in Step 2 *is* the storage (it's the container filesystem). Practically:

- `/workspace` is just a directory on that disk, so every command above uses it
  unchanged.
- Data on the disk **persists while the instance exists**, including across
  **stops** (you keep paying a small storage fee while stopped). **Destroying**
  the instance deletes it.
- **Iterating on code:** edit under `/opt/pixie` on a running instance (editable
  `pip install -e` picks it up immediately). That edit lives only on that
  instance — commit anything you want to keep to your fork and rebuild the image
  (only needed when `environment.yaml` or a compiled extension changes).
- **Reusing models across *different* instances:** either re-run
  `download_models.py` on each, or attach a **Vast.ai Volume** (their newer
  persistent-volume feature) and point `paths.base_path` at its mount path so
  checkpoints and the HF cache survive independently of any one instance.

> **Match the image arch to the offer:** the `sm86-a6000` image runs only on
> A6000/A40/A5000/3090. If you can only find a **4090/L40** offer, launch your
> `sm89-4090` image instead; an A100/H100 offer needs a `multi` (or `8.0`/`9.0`)
> build.

---

## Notes & tuning

- **Build OOM (flash-attn):** the default `MAX_JOBS=6` fits a 64 GB build box
  (its nvcc jobs use ~3-6 GB each). On less RAM, lower it —
  `MAX_JOBS=4 IMAGE_REPO=... ./docker/build_and_push.sh a6000`; on a bigger box
  raise it (e.g. `MAX_JOBS=16`) for a faster build.
- **CUDA version:** the image targets CUDA 12.1 (torch 2.1.2/cu121). The host
  NVIDIA driver just needs to support CUDA ≥ 12.1; the toolkit is inside the
  image, so the Vast host's driver is fine.
- **Wrong-arch image won't launch:** an `sm86-a6000` image runs only on
  Ampere/GA102 cards. If you rent a 4090/L40 or A100/H100 offer it will fail —
  match the instance GPU to the image tag, or build the `multi` arch image.
- **Training the UNet** (`third_party/Wavelet-Generation/trainer/...`) is the
  heavy multi-GPU path (paper: 6× A6000). For that, rent a multi-GPU A6000
  instance with the `sm86-a6000` image and scale `training.training.batch_size`
  to your card count; the physical-parameter-estimation MVP above does **not**
  need it.
