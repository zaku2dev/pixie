# Building the Pixie image on an AWS EC2 instance

The Pixie image must be built on a **native x86-64 (amd64) Linux host with a
Docker daemon** — it does **not** need a GPU (the CUDA arch is baked in via a
build arg). A short-lived EC2 CPU instance is a clean way to do this. Spin it
up, build, push to Docker Hub, terminate.

> Before you start: make sure your latest commit (the one containing
> `Dockerfile`, `docker/`, `.dockerignore`) is **pushed** to your fork's branch
> — EC2 will `git clone` it. From your laptop: `git push origin main`.

---

## 1. Launch the instance

EC2 console → **Launch instance**:

- **AMI:** *Ubuntu Server 22.04 LTS*, architecture **64-bit (x86)** — **not**
  the Arm/Graviton variant. (RunPod GPUs are x86; the image must be x86.)
- **Instance type:** **`c7i.4xlarge`** (16 vCPU / 32 GB RAM) — recommended, with
  `MAX_JOBS=4` (see step 7). flash-attn — which would otherwise be the one
  memory-hungry compile — is installed as a **prebuilt wheel** (Dockerfile
  STEP 7), so 64 GB is unnecessary;
  what still compiles from source (tiny-cuda-nn, PyTorch3D, the gaussian
  rasterizer, simple-knn) fits comfortably in 32 GB. Bigger/faster:
  `m7i.4xlarge` (16 vCPU / 64 GB) — more headroom, raise `MAX_JOBS` for a quicker
  build. Smaller/cheaper: `c7i.2xlarge` (8 vCPU / 16 GB) with `MAX_JOBS=2` works
  but roughly doubles build time (fewer cores) and leaves little RAM headroom.
  Tick **Spot** for ~⅓ the price on a throwaway build box.
- **Key pair:** select or create one for SSH.
- **Network / security group:** allow inbound **SSH (TCP 22)** from *My IP*.
- **Storage:** change the root volume to **120 GB gp3** (the image + layers +
  build cache are large; the 8 GB default will fill up).

Launch.

## 2. SSH in

```bash
ssh -i /path/to/your-key.pem ubuntu@<instance-public-ip>
```

## 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker                 # apply the group without re-login
docker run --rm hello-world   # sanity check
```

(No NVIDIA Container Toolkit needed — the build never runs the GPU.)

## 4. Clone your fork

```bash
git clone -b main https://github.com/zaku2dev/pixie.git
cd pixie
```

No `--recurse-submodules` needed — `third_party/` (including the PhysGaussian
`simple-knn` / `diff-gaussian-rasterization` CUDA sources) is vendored as
regular tracked files.

## 5. Provide the GitHub token for the in-image clone

The image **git-clones the code inside the build** (so the container is a live
repo you can `git pull` later), and the repo is private — so the build needs a
GitHub token. `build_and_push.sh` reads it from `GITHUB_TOKEN` and forwards it as
a BuildKit **secret**, so it is never written into an image layer.

Create a **fine-grained PAT** scoped to just this repo — **Contents: read-only**,
short expiration — at github.com/settings/tokens. Then, on the EC2 box, read it
into the environment **without** putting it in shell history or the EBS volume:

```bash
read -rsp 'GitHub token: ' GITHUB_TOKEN; export GITHUB_TOKEN; echo
```

- Typed via `read -s`, the token stays only in the shell's memory — not in
  `~/.bash_history`, not on disk — and is gone when you terminate the instance.
- Do **not** `export GITHUB_TOKEN=ghp_...` inline (it lands in history and `ps`),
  and do **not** pass it via EC2 user-data (readable by anything on the box via
  IMDS).

## 6. Log in to Docker Hub

Create an access token first: hub.docker.com → **Account Settings → Security →
New Access Token** (read/write). Then:

```bash
docker login -u zaku2dev
# paste the ACCESS TOKEN as the password (not your account password)
```

## 7. Build and push

```bash
# On the recommended c7i.4xlarge (32 GB), cap parallelism to fit RAM:
MAX_JOBS=4 IMAGE_REPO=zaku2dev/pixie ./docker/build_and_push.sh a6000   # -> :sm86-a6000
# or, for RTX 4090 pods:
MAX_JOBS=4 IMAGE_REPO=zaku2dev/pixie ./docker/build_and_push.sh 4090    # -> :sm89-4090
```

- The script already forces `--platform linux/amd64` (a no-op here since EC2 is
  native x86 — no emulation, unlike an Apple-Silicon Mac).
- `MAX_JOBS` caps the parallel compile jobs. flash-attn — which would otherwise be
  the memory-hungry one — installs as a **prebuilt wheel** (not compiled), so
  `MAX_JOBS` only governs the lighter source compiles (tiny-cuda-nn, PyTorch3D,
  gaussian rasterizer, simple-knn). Use **`4` on 32 GB**, `2` on a 16 GB box, and
  raise it (e.g. `16`) on a 64 GB box for a faster build.
- Expect **~30–60 min** on 16 vCPU (tiny-cuda-nn, PyTorch3D and the gaussian
  rasterizer compile from source; flash-attn is a wheel). Fewer vCPUs scale the
  time up roughly linearly.

## 8. Verify the push

Check `https://hub.docker.com/r/zaku2dev/pixie/tags` — you should see the
`sm86-a6000` (and/or `sm89-4090`) tag. That image is now what you point your
RunPod template at (see [DOCKER_RUNPOD.md](../DOCKER_RUNPOD.md) Step 3, or
[DOCKER_VASTAI.md](../DOCKER_VASTAI.md) for Vast.ai).

## 9. Terminate the instance (stop billing)

EC2 console → **Instances** → select → **Instance state → Terminate**. The
root EBS volume is set to *delete on termination* by default; confirm it's gone
so you're not billed for idle storage.

---

### Cost sketch

`c7i.4xlarge` is ~\$0.71/hr on-demand (less on Spot; `m7i.4xlarge` ~\$0.80/hr). A
single build run is on the order of **\$1**. You only pay while the instance is
running, so terminate as soon as the push completes — the built image lives in
Docker Hub, not on EC2.

### If you need both arches

You can build both in one EC2 session, but be aware it's roughly **two full
builds**, not one-plus-a-bit:

```bash
IMAGE_REPO=zaku2dev/pixie ./docker/build_and_push.sh a6000
IMAGE_REPO=zaku2dev/pixie ./docker/build_and_push.sh 4090
```

`TORCH_CUDA_ARCH_LIST` is set as an `ENV` near the top of the Dockerfile, so
changing the arch invalidates the Docker cache for *every* layer after it
(including apt/conda/torch), and the second preset recompiles from scratch.
Budget the time/cost accordingly. (If you'll rebuild multiple arches often, the
Dockerfile could be refactored to pass the arch only to the compile `RUN`s so
the arch-independent layers are shared — ask and I'll do it.)
