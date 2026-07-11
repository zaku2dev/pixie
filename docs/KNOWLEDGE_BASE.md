# Pixie — Repository Knowledge Base

> Auto-generated knowledge base combining a scan of this repository with the paper
> **"Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels"**
> (arXiv:[2508.17437](https://arxiv.org/abs/2508.17437)).
>
> Project page: https://pixie-3d.github.io/ · GitHub (upstream): `vlongle/pixie`

---

## 1. What Pixie is (the paper in one page)

**Problem.** Photorealistic 3D reconstructions (NeRF, Gaussian Splatting) capture geometry
and appearance but contain **no physics**, so reconstructed scenes are static and cannot be
simulated. Prior methods (e.g. PhysGaussian, PhysDreamer, and other "physics-from-video"
works) recover physical material fields via **slow, per-scene test-time optimization** that is
not reusable across scenes.

**Idea.** Pixie reframes physical-property estimation as **supervised, feed-forward learning**.
A neural network maps **pretrained visual features (CLIP)**, lifted into a 3D voxel grid, to a
**dense material field** of physical properties — in a single forward pass. This makes physics
inference **fast** (orders of magnitude faster than optimization) and **generalizable** across
scenes, and it **zero-shot transfers to real scenes** despite training only on synthetic data.

**Method (high level).**
1. Reconstruct the scene and distill CLIP features into 3D (F3RM-style CLIP feature field on
   top of a NeRF).
2. Voxelize into a feature grid (default `64³`, 768-dim CLIP features per occupied voxel).
3. A **3D U-Net** predicts, per voxel, a **material field**: continuous physical parameters
   plus a discrete material class.
4. Map predictions back to world coordinates, interpolate onto the Gaussian Splatting point
   cloud, and simulate with an **MPM (Material Point Method)** solver (PhysGaussian).

**Predicted physical quantities (per voxel):**
- `density` (ρ)
- `E` — Young's modulus (learned/stored in **log** space)
- `nu` (ν) — Poisson's ratio
- `material_id` — discrete material class (one of **8 classes**, id `7` = background;
  ids `0–6` are foreground material types)

**Dataset — PixieVerse.** One of the largest known datasets of paired 3D assets + physical
material annotations, mined from Objaverse and auto-labeled by a **VLM labeling pipeline**
(with a VLM "critic" + light manual correction). Stats from
`normalization_stats/material_statistics.json`: **1,577 objects**, ~**413M voxels**; heavily
background-dominated (~87% of voxels are background id `7`).

**Headline results (from paper).** Pixie beats test-time optimization baselines by
**1.46×–4.39×** on the relevant metric while being **orders of magnitude faster**, and
generalizes zero-shot to real captured scenes.

**Authors.** Long Le¹, Ryan Lucas², Chen Wang¹, Chuhao Chen¹, Dinesh Jayaraman¹, Eric Eaton¹,
Lingjie Liu¹. ¹University of Pennsylvania · ²MIT.

```bibtex
@article{le2025pixie,
  title={Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels},
  author={Le, Long and Lucas, Ryan and Wang, Chen and Chen, Chuhao and Jayaraman, Dinesh and Eaton, Eric and Liu, Lingjie},
  journal={arXiv preprint arXiv:2508.17437},
  year={2025}
}
```

---

## 2. End-to-end pipeline (`pipeline.py`)

`pipeline.py` is the Hydra entry point that orchestrates everything. It runs the same stages
for both **data generation** (`material_mode=vlm`) and **inference** (`material_mode=neural`);
the two modes differ only in *how the material field is produced* in step 6.

| # | Stage | Code | Notes |
|---|-------|------|-------|
| 1 | Download asset | `download_assets` → `download_object` | Objaverse by `obj_id` (skipped if `obj_path` given or `is_objaverse_object=false`). |
| 2 | Render images | `render_blender_images` | Blender + BlenderNeRF add-on, `data_rendering.num_images` (default 200) views. Objaverse only. |
| 3 | Train CLIP-distilled NeRF | `train_distilled_clip_nerf` | Uses **F3RM** via `ns-train` (nerfstudio). Produces a distilled CLIP feature field. |
| 4 | Train Gaussian Splatting | `train_gaussian_splatting` | PhysGaussian's gaussian-splatting `train.py`. Static appearance/geometry for later simulation. |
| 5 | Voxelize | `generate_voxels` → `pixie/voxel/voxelize.py` | Samples the CLIP field into a `grid_size³` (default 64) voxel feature grid → `clip_features.npz` (+ `clip_features_pc.ply`). Scene bounds `[-0.5, 0.5]³`. |
| 6 | Material field | `material_mode` branch | **vlm**: VLM labeling pipeline (below) → `generate_material_segmentation`. **neural**: `generate_neural_segmentation` runs the trained 3D U-Net. |
| 7 | Physics simulation | `run_physics_simulation` | Maps predicted grid → world coords → interpolates onto GS point cloud → runs `gs_simulation.py` (PhysGaussian MPM). Renders frames + video. |

**`render.py`** is a separate, optional "fancy render" step in Blender for presentation-quality
simulation videos.

### Two operating modes
- **`material_mode=vlm`** — how PixieVerse is *generated*. A VLM proposes part segmentations and
  physical parameters using in-context tuned examples. Used as **training labels**.
- **`material_mode=neural`** — how Pixie *infers* physics. The trained 3D U-Net predicts the
  material field directly from CLIP features. This is the actual "physics from pixels" model.

---

## 3. VLM labeling pipeline (data generation)

Lives in `pixie/vlm_labeler/`. Invoked from `pipeline.py` only when `material_mode=vlm` and
`segmentation.vlm.labeling.enabled=true`. Sequence:

1. `vlm_seg.py` — **segmentation**: VLM proposes part queries (default 5 alternative queries,
   15 input views) → `vlm_results.json`.
2. `vlm_viz_seg_candidates.py` — renders/visualizes each candidate CLIP-based segmentation.
3. `vlm_seg_critic.py` — a stronger VLM **critic** scores segmentation candidates (10 views).
4. `vlm_phys_sampler.py` — **samples physical parameters** (E, density, ν) per part from a
   plausible range (`num_sample_mat` samples).
5. `vlm_parse_seg_critic.py` — parses/finalizes the chosen result →
   `chosen_vlm_results.json` (contains the per-part `material_dict`).

Other labelers: `vlm_phys_judge.py`, `vlm_parse_seg_critic.py`, `vlm_seg_class_instruction.py`
(per-class in-context instructions), `vlm_data_filtering.py` (asset quality filter).

**VLM backends** are configurable in `config/segmentation/default.yaml` — OpenAI (GPT),
Anthropic (Claude), Google Gemini (default across stages), or **Qwen2.5-VL** (local, no API
key). Default models are Gemini variants (`gemini-2.5-pro`, `gemini-2.0-flash`, etc.). API keys
come from config or `.env`.

### Data curation (`data_curation/`) — mining Objaverse
1. `objaverse_selection.py` — cosine similarity between Objaverse object names and each target
   class, keep top-k.
2. `download_objaverse.py` — download selected assets.
3. `render_objaverse_classes.py` — render 1 view/object for filtering.
4. `vlm_data_filtering.py` — VLM filters low-quality assets.
5. `manual_data_filtering_correction.py` — Streamlit UI to correct VLM filtering →
   `all_results_corrected.json`.
6. `manual_sim_validation.py` — Streamlit UI to validate simulation videos.

`config/obj_ids_metadata.json` is the precomputed manifest of `obj_id → {obj_class,
vlm_filtering.is_appropriate}`. `data_curation/cat_dict.json` lists the semantic classes.

**Object classes** (from `cat_dict.json`): tree, flowers, bread, water-like_bodies, pillows,
rubber_ducks_and_toys, soda_cans, sport_balls, jello_block, sand, shrubs, rocks, metal_crates,
barrels, plastic_bottles, statues, couches, chairs, grass, cars, lamps, fruit, snow_and_mud.
(The paper describes ~10 headline semantic classes used for the main experiments.)

---

## 4. The learned model (3D U-Net)

> **★ Key component for physical-parameter estimation.**
> The piece that actually *estimates the numeric physical properties* is the **continuous 3D
> regression U-Net — `RegressionUNet`** (`third_party/Wavelet-Generation/trainer/training_continuous_mse.py`),
> run at inference by `inference_combined.py`. It takes the voxelized **CLIP feature grid**
> (768-d/voxel) and outputs a dense per-voxel field with **`out_channels=3` → density, E
> (log Young's modulus), ν (Poisson's ratio)** (`inference_combined.py:102`).
>
> Its companion **`SegmentationUNet`** (`training_discrete.py`) predicts the discrete **material
> class id** (8 classes, `argmax` over logits) — material *type*, not the numeric parameters.
> `inference_combined.py` runs both on the same feature grid and stacks the results:
> ```python
> seg_logits = seg_network(feat_grid)          # discrete material class
> seg_pred   = torch.argmax(seg_logits, dim=1)
> cont_pred  = cont_network(feat_grid)         # density, E, nu  ← physical parameters
> ```
> Outputs are then denormalized with the log/percentile ranges in `normalization_stats/`.
> Pipeline path: `pipeline.py` (`material_mode=neural`) → `generate_neural_segmentation`
> (`pixie/utils.py:724`) → `inference_combined.py`.
>
> Note on provenance: the U-Net *learns* to predict these values, but the **ground-truth labels
> it trains on are produced by the VLM physics sampler** (`vlm_phys_sampler.py`, §3). So the VLM
> is the ultimate *source* of parameter values; `RegressionUNet` is what makes estimation fast,
> feed-forward, and generalizable at inference time.

Training/inference code lives in `third_party/Wavelet-Generation/` (a wavelet-generation
codebase repurposed as the 3D backbone). Config: `config/training/default.yaml`.

**Inputs / outputs.**
- Input: voxel **feature grid**, `feature_type ∈ {clip(768-d), rgb(3-d), occupancy(1-d)}`
  (default `clip`), `cond_dim=32`.
- Output: per-voxel **material field** with `in_material_channels=4`
  (density, E, nu, material_id). Continuous head predicts `num_cont_channels=3`
  (density, log-E, ν); discrete head predicts `num_material_classes=8` (one-hot; class 7 =
  background).

**Two networks** (trained separately, combined at inference):
- **Continuous** — regresses density/log-E/ν (`training_continuous_mse.py`, MSE loss).
- **Discrete** — classifies material id (`training_discrete.py`).
- `inference_combined.py` merges both at test time.

**Backbone / training.** 3D U-Net (`unet_model_channels=64`, `num_res_blocks=3`,
`channel_mult=[1,1,2,4]`), instance norm, `bior6.8` wavelet, resolution 256. Adam, lr `1e-4`
with decay, `batch_size=4`, `training_epochs=300`, `train_size=0.9`, seed 69. Loss weights
`lambda_cont=1.0`, `lambda_cat=2.0`. Optional DDIM/diffusion machinery is present (inherited
from Wavelet-Generation) but the primary continuous model is trained with **MSE**.
`enforce_mask_consistency=true` keeps continuous predictions consistent with the occupancy mask.

**Hardware used (paper/README).** 6× NVIDIA RTX A6000 (~49 GB), 128 CPUs, 450 GB RAM per model.

### Normalization (critical before training)
Physical parameters span many orders of magnitude, so density and E are handled in **log**
space and normalized to learned p1–p99 percentile ranges. Ranges are **data-driven** — there
are no hardcoded fallbacks. Run `third_party/Wavelet-Generation/data_utils/inspect_ranges.py`
to (re)compute them into `normalization_stats/normalization_ranges.yaml`.

Current stats (`normalization_stats/`):
- `density`: log p1–p99 ≈ **1.70 – 3.87**
- `E` (Young's modulus): log p1–p99 ≈ **3.02 – 10.88**
- `nu` (Poisson's ratio): p1–p99 ≈ **0.210 – 0.449**
- Raw ranges are huge (E up to ~2×10¹¹ Pa), motivating log-space handling.
- `material_statistics.json`: 1,577 objects, ~413M voxels, ~87% background.

---

## 5. Voxelization & mapping back to world

- `pixie/voxel/voxelize.py` — samples the trained CLIP feature field into a dense grid.
  Key knobs (`config/voxelization/default.yaml`): `grid_size=64`, `gray_threshold=0.05`
  (drops near-gray/empty voxels), scene bounds `[-0.5,0.5]³`, `alpha_weighted=true`,
  `alpha_threshold_for_mask=0.01` for occupancy.
- `pixie/voxel/segmentation.py`, `viz_segmentation.py` — CLIP-feature-based part segmentation
  and visualization.
- `pixie/voxel/map_pred_to_coords.py` — maps predicted voxel grid → world coordinates and
  interpolates onto the Gaussian Splatting point cloud so MPM can simulate it. For Objaverse,
  NeRF frame == world frame; for **real scenes** a `dataparser_transforms.json` (nerfstudio)
  is needed to align frames (`config/mapping/default.yaml`).

---

## 6. Physics simulation (PhysGaussian / MPM)

- Runs `gs_simulation.py` inside `third_party/PhysGaussian` via `run_physics_simulation`.
- Inputs: trained GS model + segmented material point cloud + a **physics config** (per
  obj/class/mode, resolved by `get_physics_config_path`).
- `config/physics/default.yaml`: `save_ply` (per-frame PLY, slower — only for fancy Blender
  renders), `white_bg` (auto, except `no_white_bg_classes=["snow_and_mud"]`), `sample_id`,
  `debug`. Runs headless via `xvfb-run`, renders frames, compiles video (`output.mp4`/`.gif`).

---

## 7. Repository map

```
pixie/                     # Main package
  utils.py                 # pipeline glue: paths, downloads, seg dispatch, config resolution
  training_utils.py        # training helpers
  metrics.py               # eval metrics + material-statistics aggregation from VLM results
  viz_utils.py             # visualization helpers
  blender/                 # Blender rendering (BlenderNeRF data gen, GLB/GS render, feature colors)
  vlm_labeler/             # VLM-based segmentation + physical-parameter labeling (data gen)
  voxel/                   # voxelization, CLIP segmentation, map-pred-to-world-coords
pipeline.py                # MAIN Hydra entry point (full pipeline; vlm | neural)
render.py                  # optional fancy Blender render of a simulation
config/                    # Hydra config groups (see §8)
data_curation/             # Objaverse mining, rendering, VLM + manual filtering, sim validation
normalization_stats/       # normalization_ranges.yaml, material_statistics.json
scripts/                   # download/upload models & data (HuggingFace)
nbs/                       # pixie.ipynb (synthetic), real_scene.ipynb (result inspection)
docs/                      # assets, fonts, THIS knowledge base
third_party/               # PhysGaussian, Wavelet-Generation (3D U-Net), f3rm, nerfstudio, vlmx
```

### Third-party dependencies (vendored in `third_party/`)
- **nerfstudio** — NeRF training framework (`ns-train`).
- **f3rm** — distills CLIP features into a 3D feature field on top of the NeRF.
- **Wavelet-Generation** — houses the **3D U-Net** training/inference (Pixie's learned model).
- **PhysGaussian** — Gaussian Splatting + **MPM** physics solver (`gs_simulation.py`).
- **vlmx** — VLM utility wrappers used by the labelers.

---

## 8. Configuration (Hydra)

Root `config/config.yaml` composes these groups (each has a `default.yaml`):
`paths`, `data_rendering`, `output_rendering`, `training_3d`, `training`, `voxelization`,
`segmentation`, `physics`, `data_curation`, `mapping`.

Key top-level flags:
- `obj_id` (required for Objaverse) / `obj_path` (local file) / `obj_class` (auto-detected).
- `material_mode: vlm | neural` (default `vlm`).
- `is_objaverse_object: true|false` (set `false` for real scenes).
- `overwrite`, `overwrite_voxel` — force re-run of cached stages.

Override anything on the CLI, Hydra-style: `python pipeline.py obj_id=... material_mode=neural`.
`config/paths/default.yaml` holds machine-specific paths (Blender path, add-on zips,
PhysGaussian/GS dirs, data/output roots). `config/training_3d/default.yaml` controls NeRF & GS
iteration counts.

---

## 9. Data layout (PixieVerse) — see `data_readme.md`

Rooted at Hydra `paths.base_path`. Top-level folders:
`data/` (per-object rendered views), `outputs/` (reconstruction/training runs),
`render_outputs/` (final sim/render artifacts, `sample_*/gs_sim_gridsize_<D>_output/`),
`vlm_seg_results/`, `vlm_seg_critic_results/`, `vlm_seg_mat_sample_results/`
(`sample_*/chosen_vlm_results.json`).

Downloadable from HuggingFace instead of regenerating:
- Dataset: `vlongle/pixieverse` — `python scripts/download_data.py --dataset-repo vlongle/pixieverse --dirs archives --local-dir <root>` (add `--obj-class tree` for a quick single-class test), then unpack the `archives/*.tar` into the standard folders (script in README/`data_readme.md`).
- Model checkpoints: `vlongle/pixie` — `python scripts/download_models.py`.

---

## 10. How to run (quick reference)

```bash
# Inference on an Objaverse object with the trained neural model
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 material_mode=neural

# Generate labels for that object (VLM mode — how PixieVerse is built)
python pipeline.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8 material_mode=vlm

# Real captured scene (no Blender render/download; needs colmap + dataparser transform)
python pipeline.py is_objaverse_object=false obj_id=bonsai material_mode=neural \
  paths.data_dir='${paths.base_path}/real_scene_data' \
  paths.outputs_dir='${paths.base_path}/real_scene_models' \
  paths.render_outputs_dir='${paths.base_path}/real_scene_render_outputs' \
  training.enforce_mask_consistency=false

# Fancy Blender render of the simulation
python render.py obj_id=f420ea9edb914e1b9b7adebbacecc7d8
```

**Training the model** (after downloading/generating PixieVerse):
```bash
python third_party/Wavelet-Generation/data_utils/inspect_ranges.py           # 1. normalization
python third_party/Wavelet-Generation/trainer/training_discrete.py           # 2a. discrete
python third_party/Wavelet-Generation/trainer/training_continuous_mse.py     # 2b. continuous
python third_party/Wavelet-Generation/trainer/inference_combined.py [obj_id=...]  # 3. inference
```

Inspect outputs in `nbs/pixie.ipynb` (synthetic) or `nbs/real_scene.ipynb` (real).

---

## 11. Gotchas & environment notes

- **Install order matters** (see README §Installation): torch/torchvision matched to CUDA,
  then `tiny-cuda-nn`, `nerfstudio`, `f3rm`, `pytorch3d`, then PhysGaussian CUDA submodules
  (`simple-knn`, `diff-gaussian-rasterization`), then `vlmx`, `flash-attn` (for Qwen2.5-VL).
- Python 3.10, conda env `pixie`. Blender **4.3.2** with BlenderNeRF + gaussian-splatting
  add-ons; install `objaverse` into Blender's bundled python.
- Known fixes: `warp_lang==0.10.1` (UnicodeEncodeError), `numpy==1.24.4` (dtype/binary
  incompatibility), install GS submodules without `-e` if it fails.
- **You must run `inspect_ranges.py` before training** — no hardcoded normalization fallbacks.
- Most pipeline stages are **cached/idempotent**: they skip if outputs exist unless
  `overwrite` / `overwrite_voxel` is set.
- Simulation runs headless (`xvfb-run`) and needs a GPU; `save_ply=true` is slow (only for
  Blender render).

---

*Generated 2026-07-11 from a scan of this repo (branch `claude`) + arXiv:2508.17437.
Line/config references reflect the code at generation time; re-verify against source if the
repo has since changed.*
