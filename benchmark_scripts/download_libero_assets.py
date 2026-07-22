import init_path
import argparse
import os

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
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Place a plain copy in the assets dir instead of symlinking into "
             "the shared Hugging Face cache.",
    )
    parser.add_argument(
        "--link",
        type=str,
        default=None,
        metavar="EXISTING_DIR",
        help="Skip downloading; symlink the package assets dir to an assets "
             "tree that already exists at this path. Falls back to "
             "$LIBERO_ASSET_PATH if set.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exit without prompting if assets are already present (for "
             "non-interactive installs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_existing and download_utils.assets_are_present(args.assets_dir):
        print("LIBERO assets already present; skipping download.")
        return
    link_target = args.link or os.environ.get(download_utils.LIBERO_ASSET_PATH_ENV)
    if link_target:
        if args.link is None:
            print(f"Using {download_utils.LIBERO_ASSET_PATH_ENV}={link_target}")
        download_utils.libero_assets_link(link_target, assets_dir=args.assets_dir)
        return
    download_utils.libero_assets_download(
        assets_dir=args.assets_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        check_overwrite=not args.force,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
