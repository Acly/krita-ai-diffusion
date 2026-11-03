"""Send generated images to the Eagle app (https://eagle.cool) via its local API."""

from __future__ import annotations

import asyncio
import platform
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from . import util
from .backend.network import RequestManager
from .localization import translate as _
from .platform_tools import is_macos, is_windows
from .text import create_img_metadata
from .util import client_logger as log

if TYPE_CHECKING:
    from .model.jobs import Job

EAGLE_URL = "http://localhost:41595"

_upload_dir = util.user_data_dir / "eagle_upload"
_requests: RequestManager | None = None


async def prepare_eagle_installer(callback):
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices
    from PyQt5.QtNetwork import QNetworkAccessManager

    from .backend.network import download
    from .backend.server import InstallationProgress

    if not (is_windows or is_macos):
        raise RuntimeError("Eagle is available for Windows and macOS.")

    suffix = (
        "x64-build23.exe"
        if is_windows
        else ("arm64-build23.dmg" if platform.machine() == "arm64" else "x64-build23.dmg")
    )
    name = f"Eagle-4.0-{suffix}"
    target = util.user_data_dir / "installers" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    network = QNetworkAccessManager()
    if not target.exists():
        async for progress in download(
            network, f"https://r2-app.eagle.cool/releases/{name}", target
        ):
            callback(InstallationProgress("Downloading Eagle", progress))
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
        raise RuntimeError(f"Could not launch Eagle setup: {target}")


def _request_manager() -> RequestManager:
    # keep one instance alive - a per-call QNetworkAccessManager can be
    # garbage-collected before the reply arrives
    global _requests
    if _requests is None:
        _requests = RequestManager()
    return _requests


def _cleanup_old_uploads(max_age_hours: float = 24):
    try:
        if not _upload_dir.exists():
            return
        cutoff = time.time() - max_age_hours * 3600
        for f in _upload_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Could not clean up eagle upload dir: {e}")


async def send_image_to_eagle(image, job: Job, index: int) -> str | None:
    """Save composited result to a temp file and import it into Eagle.

    Returns an error message on failure, None on success. The temp file is kept
    (Eagle imports asynchronously) and cleaned up on later calls after 24h.
    """
    _cleanup_old_uploads()
    _upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = job.timestamp.strftime("%Y%m%d-%H%M%S")
    prompt = util.sanitize_prompt(job.params.name)
    file_name = f"krita-ai-{timestamp}-{index}-{prompt}"[:120] + ".png"
    path = util.find_unused_path(_upload_dir / file_name)

    metadata_text = create_img_metadata(job.params)
    image.save_png_with_metadata(filepath=path, metadata_text=metadata_text)

    rating = job.rating(index)
    tags = ["krita-ai"]
    if style := job.params.metadata.get("style"):
        tags.append(str(style))
    for lora in job.params.metadata.get("loras", []):
        if isinstance(lora, dict) and lora.get("name"):
            tags.append(Path(str(lora["name"])).stem)

    # A marker tag to find this exact item again below (job.params.name is not
    # unique - eg. re-running the same prompt twice - and Eagle silently
    # renames duplicate item names on import, so matching by name is unreliable)
    marker = f"kai-{uuid.uuid4().hex[:10]}"
    if rating > 0:
        tags.append(marker)

    payload = {
        "path": str(path),
        "name": job.params.name[:100] or "Krita AI generation",
        "annotation": metadata_text,
        "tags": tags,
    }
    try:
        result = await _request_manager().post(f"{EAGLE_URL}/api/item/addFromPath", payload)
        if isinstance(result, dict) and result.get("status") == "success":
            if rating > 0:
                await _apply_rating(marker, rating)
            return None
        return _("Eagle returned an error") + f": {result}"
    except Exception as e:
        log.warning(f"Could not send image to Eagle: {e}")
        return _("Could not reach Eagle - is the app running?") + f" ({e})"


async def _apply_rating(marker: str, rating: int):
    """addFromPath doesn't accept a star rating and doesn't return the new item's
    id either, so poll recently added items for the one carrying our marker tag
    and set its rating via a separate item/update call."""
    for _attempt in range(6):
        await asyncio.sleep(0.5)
        try:
            # no orderBy: default listing is already newest-first, and
            # orderBy=-CREATEDATE was observed to sort oldest-first instead
            result = await _request_manager().get(
                f"{EAGLE_URL}/api/item/list?limit=10&tags={marker}"
            )
            items = result.get("data", []) if isinstance(result, dict) else []
            match = next((i for i in items if marker in (i.get("tags") or [])), None)
            if match and match.get("id"):
                await _request_manager().post(
                    f"{EAGLE_URL}/api/item/update", {"id": match["id"], "star": rating}
                )
                return
        except Exception as e:
            log.warning(f"Could not set Eagle rating: {e}")
            return
