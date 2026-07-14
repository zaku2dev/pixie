#!/usr/bin/env python3
"""
Download PixieVerse data from Hugging Face.
"""

import argparse
import fnmatch
import os
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def download_data(
    dataset_repo: str = "vlongle/pixieverse",
    download_dirs: list[str] | None = None,
    force_download: bool = False,
    local_dir: str | None = None,
    obj_class: str | None = None,
    token: str | None = None,
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    download_path = Path(local_dir) if local_dir else project_root
    download_path.mkdir(parents=True, exist_ok=True)

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
    print(f"Available directories: {available_dirs}")

    if download_dirs:
        dirs_to_download = [d for d in download_dirs if d in available_dirs]
        missing = sorted(set(download_dirs) - set(dirs_to_download))
        if missing:
            print(f"Requested directories not found: {missing}")
    else:
        dirs_to_download = available_dirs

    if obj_class and not download_dirs:
        dirs_to_download = [d for d in ["archives"] if d in available_dirs]
        print("Using class-filter mode: defaulting to --dirs archives")

    if not dirs_to_download:
        print("No directories selected for download.")
        return

    allow_patterns = []
    if obj_class:
        assert "archives" in dirs_to_download, (
            "--obj-class filtering currently supports archive layout only. "
            "Please include --dirs archives (or omit --dirs)."
        )
        allow_patterns.extend(
            [
                f"archives/*/{obj_class}.tar",
                f"archives/*/{obj_class}.tar.gz",
            ]
        )
        available_class_files = [
            f for f in data_files if any(fnmatch.fnmatch(f, pat) for pat in allow_patterns)
        ]
        if not available_class_files:
            print(f"No archive files found for obj_class='{obj_class}'.")
            return
        print(f"Downloading class '{obj_class}' from archives only.")
    else:
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
    if obj_class:
        print(f"Downloaded archives for class '{obj_class}' to {download_path}")
    else:
        print(f"Downloaded {dirs_to_download} to {download_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PixieVerse data from Hugging Face.")
    parser.add_argument("--dataset-repo", default="vlongle/pixieverse", help="Hugging Face dataset repo id.")
    parser.add_argument("--dirs", nargs="*", help="Specific directories to download (default: all).")
    parser.add_argument("--local-dir", help="Local destination directory (default: project root).")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    parser.add_argument(
        "--obj-class",
        default=None,
        help="Download only one class archive (e.g., tree). Works with --dirs archives.",
    )
    parser.add_argument("--token", help="HuggingFace token (defaults to HF_TOKEN env var).")
    args = parser.parse_args()

    download_data(
        dataset_repo=args.dataset_repo,
        download_dirs=args.dirs,
        force_download=args.force,
        local_dir=args.local_dir,
        obj_class=args.obj_class,
        token=args.token,
    )


if __name__ == "__main__":
    main()
