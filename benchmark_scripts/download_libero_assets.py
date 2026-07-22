import init_path
import argparse

import libero.libero.utils.download_utils as download_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LIBERO simulation assets from the Hugging Face Hub."
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=None,
        help="Target directory (default: the installed package assets dir).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"HF repo id (default: {download_utils.HF_ASSETS_REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default=None,
        choices=["dataset", "model"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if assets already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_utils.libero_assets_download(
        assets_dir=args.assets_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        check_overwrite=not args.force,
    )


if __name__ == "__main__":
    main()
