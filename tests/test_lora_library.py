from unittest.mock import AsyncMock

from ai_diffusion.backend.lora_manager import LoraInfo, fetch_loras_pages

from .conftest import qtapp


@qtapp
async def test_incomplete_manager_list_includes_all_comfy_files():
    requests = AsyncMock()

    async def get(url, **kwargs):
        if "favorites_only" in url:
            return {"items": []}
        if "/api/lm/loras/list" in url:
            return {
                "items": [{"file_name": "known.safetensors", "folder": "Krea2", "sha256": "abc"}],
                "total": 1,
            }
        return ["krea2\\known.safetensors", "Krea2/missing.SAFETENSORS", "private.safetensors"]

    requests.get.side_effect = get
    loras = [lora async for batch in fetch_loras_pages(requests, "http://server") for lora in batch]
    assert [l.name for l in loras] == ["Krea2/known", "Krea2/missing", "private"]
    assert loras[0].sha256 == "abc"
    assert all(not l.preview_url for l in loras)


@qtapp
async def test_unavailable_manager_keeps_standard_library():
    requests = AsyncMock()

    async def get(url, **kwargs):
        if "/api/lm/" in url:
            raise RuntimeError("unavailable")
        return ["private.safetensors"]

    requests.get.side_effect = get
    loras = [lora async for batch in fetch_loras_pages(requests, "http://server") for lora in batch]
    assert [l.name for l in loras] == ["private"]


def test_manager_folder_is_not_duplicated():
    assert (
        LoraInfo.from_api({"file_name": "Krea2/model.safetensors", "folder": "Krea2"}, "").name
        == "Krea2/model"
    )
