#!/usr/bin/env python3
"""
Download model-related directories from a Hugging Face dataset repo.
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def seed_normalization_stats(project_root: Path, download_path: Path) -> None:
    """Copy the repo's canonical normalization_stats/ into the download target.

    Inference (the NEURAL_INFERENCE pipeline step) reads normalization ranges
    from ``{base_path}/normalization_stats/normalization_ranges.yaml``. These
    ranges must match what the pretrained model was trained with, so we seed the
    repo's checked-in copy rather than recompute per-dataset via inspect_ranges.py
    (which would misnormalize predictions). No-op when downloading into the repo
    itself (source == target).
    """
    src = project_root / "normalization_stats"
    dst = download_path / "normalization_stats"
    if src.resolve() == dst.resolve():
        return
    if not src.is_dir():
        print(f"Warning: {src} not found; skipping normalization_stats seeding.")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
    print(f"Seeded normalization stats into: {dst}")


def download_models(
    dataset_repo: str = "vlongle/pixie",
    download_dirs: list[str] | None = None,
    force_download: bool = False,
    local_dir: str | None = None,
    token: str | None = None,
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    download_path = Path(local_dir) if local_dir else project_root
    download_path.mkdir(parents=True, exist_ok=True)

    # Seed normalization stats into base_path so NEURAL_INFERENCE can find them
    # regardless of which model dirs are downloaded.
    seed_normalization_stats(project_root, download_path)

    # Anonymous downloads are aggressively rate-limited (HTTP 429). Authenticating
    # with a token — even a free read token for this public repo — raises the limit.
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Warning: no HF token found (set HF_TOKEN or pass --token). Downloading "
            "anonymously, which HuggingFace rate-limits and may fail with HTTP 429."
        )

    repo_files = list_repo_files(repo_id=dataset_repo, repo_type="dataset", token=token)
    data_files = [f for f in repo_files if f != "README.md" and not f.startswith(".")]
    available_dirs = sorted({f.split("/")[0] for f in data_files if "/" in f})

    if download_dirs:
        dirs_to_download = [d for d in download_dirs if d in available_dirs]
        missing = sorted(set(download_dirs) - set(dirs_to_download))
        if missing:
            print(f"Requested directories not found: {missing}")
    else:
        dirs_to_download = available_dirs

    if not dirs_to_download:
        print("No directories selected for download.")
        return

    allow_patterns = []
    for dir_name in dirs_to_download:
        allow_patterns.extend([f"{dir_name}/*", f"{dir_name}/**/*"])

    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=download_path,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=["README.md", ".gitattributes"],
        force_download=force_download,
        token=token,
    )

    print(f"Download complete to: {download_path}")
    print(f"Downloaded directories: {dirs_to_download}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pixie model artifacts from Hugging Face.")
    parser.add_argument("--dataset-repo", default="vlongle/pixie", help="Hugging Face dataset repo id.")
    parser.add_argument("--dirs", nargs="*", help="Specific directories to download (default: all).")
    parser.add_argument("--local-dir", help="Local directory to download into (default: project root).")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    parser.add_argument("--token", help="HuggingFace token (defaults to HF_TOKEN env var).")
    args = parser.parse_args()

    download_models(
        dataset_repo=args.dataset_repo,
        download_dirs=args.dirs,
        force_download=args.force,
        local_dir=args.local_dir,
        token=args.token,
    )


if __name__ == "__main__":
    main()
