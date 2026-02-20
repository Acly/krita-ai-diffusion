import json
from itertools import chain

import ai_diffusion.resources as res
from ai_diffusion.resources import Arch, ModelResource

from .config import result_dir


def test_resources_json():
    result = {}
    result["required"] = ModelResource.as_list(res.required_models)
    result["checkpoints"] = ModelResource.as_list(res.default_checkpoints)
    result["upscale"] = ModelResource.as_list(res.upscale_models)
    result["optional"] = ModelResource.as_list(res.optional_models)
    result["prefetch"] = ModelResource.as_list(res.prefetch_models)
    result["deprecated"] = ModelResource.as_list(res.deprecated_models)

    original = res._models_file.read_text()
    result_string = json.dumps(result, indent=2)
    (result_dir / "resources.json").write_text(result_string)
    assert result_string == original


def test_json_no_duplicates():
    d = json.load(res._models_file.open())
    all_ids = set()
    all_names = set()
    for cat in d.values():
        for m in cat:
            assert m["name"] not in all_names, f"Duplicate model name: {m['name']}"
            all_names.add(m["name"])
            ids = m["id"]
            ids = [ids] if isinstance(ids, str) else ids
            for id in ids:
                assert id not in all_ids, f"Duplicate model id: {id}"
            all_ids.update(ids)


def test_same_name_same_model():
    for m in res.all_models(include_deprecated=True):
        for o in res.all_models(include_deprecated=True):
            if m is not o and m.name == o.name:
                assert all(mf.url == of.url for mf, of in zip(m.files, o.files))


def test_resource_ids_exist():
    ids = chain(res.required_resource_ids, res.recommended_resource_ids)
    for resource_id in ids:
        if resource_id.arch in (Arch.sd3, Arch.qwen, Arch.qwen_e, Arch.qwen_e_p, Arch.flux2_9b):
            continue  # no model downloads yet
        model = res.find_resource(resource_id)
        assert model is not None, f"Resource ID {resource_id} not found"
