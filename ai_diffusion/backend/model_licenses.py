from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..settings import settings

bundle = Path(__file__).parent.parent / "licenses" / "krea2-v2"
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
revision: str = manifest["revision"]
model_ids = frozenset(("lora-identity_edit-krea2", "lora-general_edit-krea2"))
model_filenames = frozenset((
    "krea2_identity_edit_v2.safetensors",
    "krea2_general_edit_v2.safetensors",
))
summary = (
    "Krea 2 Identity Edit V2 and General Edit V2 are free for non-commercial use. "
    "Commercial use of the models or their outputs requires a separate written license "
    "from Lars Bouaraba. The Krea 2 Community License also applies."
)
terms_url = manifest["files"][0]["url"]
community_url = manifest["files"][1]["url"]
notice_url = manifest["files"][2]["url"]


def requires_acceptance(packages: list[str]):
    return bool(model_ids.intersection(packages))


def accepted():
    return settings.krea2_v2_license_revision == revision


def acknowledge():
    settings.krea2_v2_license_revision = revision
    settings.krea2_v2_license_accepted_at = datetime.now(timezone.utc).isoformat()
    settings.save()


def require_accepted(packages: list[str]):
    if requires_acceptance(packages) and not accepted():
        raise RuntimeError(
            "Review and accept the Krea 2 Edit V2 model terms in Managed Server setup before downloading these weights."
        )


def install_documents(base: Path, *, accepted_at: str = "", method: str = "Krita setup dialog"):
    destination = base / "models" / "loras" / "krea2-edit-v2-licenses"
    contents = []
    for document in manifest["files"]:
        data = (bundle / document["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != document["sha256"]:
            raise RuntimeError(f"Bundled model license failed verification: {document['name']}")
        contents.append((document["name"], data))
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in contents:
        target = destination / name
        if not target.exists() or target.read_bytes() != data:
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_bytes(data)
            temp.replace(target)
    receipt = {
        "model_terms_revision": revision,
        "acknowledged_at": accepted_at or datetime.now(timezone.utc).isoformat(),
        "method": method,
        "documents": manifest["files"],
        "notice": "Acknowledgement is not a commercial license grant.",
    }
    (destination / "ACKNOWLEDGEMENT.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return destination
