#!/usr/bin/env python3
"""
Script to download and upload images from/to Cloudflare R2.

Usage:
    python images.py [download|upload]
"""

import argparse
import hashlib
import sys
import typing
from pathlib import Path

import boto3
import requests
from botocore.exceptions import ClientError

root_dir = Path(__file__).parent.parent
image_dir = root_dir / "tests" / "images"

if not typing.TYPE_CHECKING:
    from service.pod.lib.environment import Config

    config = Config.from_env()
else:
    config: typing.Any = None  # type: ignore

MANIFEST_FILE = image_dir / "manifest.txt"
BASE_URL = "https://lfs.interstice.cloud"
BUCKET_NAME = "lfs"
REPO_NAME = "krita-ai-diffusion"


def compute_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_key(file_path: Path, sha256: str) -> str:
    rel_path = file_path.relative_to(image_dir).as_posix()
    return f"{REPO_NAME}/{rel_path}/{sha256}"


def compute_url(file_path: Path, sha256: str) -> str:
    return f"{BASE_URL}/{compute_key(file_path, sha256)}"


def load_manifest() -> dict[str, str]:
    manifest = {}
    if not MANIFEST_FILE.exists():
        return manifest

    with open(MANIFEST_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                local_filepath = parts[0]
                sha256 = parts[1]
                manifest[local_filepath] = sha256

    return manifest


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        f.writelines(
            f"{local_filepath} {sha256}\n" for local_filepath, sha256 in sorted(manifest.items())
        )


def download_images() -> None:
    manifest = load_manifest()

    if not manifest:
        print("No manifest file found or manifest is empty.")
        return

    print(f"Downloading {len(manifest)} images...")
    success_count = 0
    skip_count = 0
    error_count = 0

    for local_filepath, expected_sha256 in manifest.items():
        local_path = image_dir / local_filepath

        if local_path.exists():
            actual_sha256 = compute_sha256(local_path)
            if actual_sha256 == expected_sha256:
                print(f"✓ {local_filepath} (already up-to-date)")
                skip_count += 1
                continue
            else:
                print(f"⟳ {local_filepath} (outdated, re-downloading)")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = compute_url(local_path, expected_sha256)
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                f.write(response.content)

            actual_sha256 = compute_sha256(local_path)
            if actual_sha256 == expected_sha256:
                print(f"✓ {local_filepath} (downloaded)")
                success_count += 1
            else:
                print(f"✗ {local_filepath} (SHA256 mismatch after download)")
                error_count += 1
                local_path.unlink()

        except Exception as e:
            print(f"✗ {local_filepath} (download failed: {e})")
            error_count += 1
            if local_path.exists():
                local_path.unlink()

    print(
        f"\nDownload complete: {success_count} downloaded, {skip_count} skipped, {error_count} errors"
    )


def upload_images() -> None:
    manifest = load_manifest()

    if not image_dir.exists():
        print(f"Images directory does not exist: {image_dir}")
        return

    local_files = {}
    for file_path in image_dir.rglob("*"):
        if file_path.is_file() and file_path.name != "manifest.txt":
            rel_path = file_path.relative_to(image_dir).as_posix()
            local_files[rel_path] = file_path

    print(f"Found {len(local_files)} local image files")

    r2_client = boto3.client(
        "s3",
        endpoint_url=f"https://{config.secrets.cf_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=config.secrets.r2_access_key_id,
        aws_secret_access_key=config.secrets.r2_secret_access_key,
        region_name=config.secrets.r2_region,
    )

    upload_count = 0
    skip_count = 0
    error_count = 0
    new_files = {}

    for rel_path, file_path in sorted(local_files.items()):
        sha256 = compute_sha256(file_path)

        # Check if this file already exists in manifest with same SHA256
        if rel_path in manifest:
            _, manifest_sha256 = manifest[rel_path]
            if sha256 == manifest_sha256:
                print(f"✓ {rel_path} (already up-to-date)")
                skip_count += 1
                continue
            else:
                print(f"⟳ {rel_path} (updating, SHA256 changed)")

        try:
            with open(file_path, "rb") as f:
                r2_client.put_object(
                    Bucket=BUCKET_NAME,
                    Key=compute_key(file_path, sha256),
                    Body=f,
                    ContentType="application/octet-stream",
                )

            print(f"✓ {rel_path} (uploaded)")
            new_files[rel_path] = sha256
            upload_count += 1

        except ClientError as e:
            print(f"✗ {rel_path} (upload failed: {e})")
            error_count += 1
        except Exception as e:
            print(f"✗ {rel_path} (error: {e})")
            error_count += 1

    manifest.update(new_files)
    save_manifest(manifest)

    print(f"\nUpload complete: {upload_count} uploaded, {skip_count} skipped, {error_count} errors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and upload images from/to Cloudflare R2")
    parser.add_argument(
        "action",
        nargs="?",
        default="download",
        choices=["download", "upload"],
        help="Action to perform (default: download)",
    )

    args = parser.parse_args()

    try:
        if args.action == "download":
            download_images()
        elif args.action == "upload":
            upload_images()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
