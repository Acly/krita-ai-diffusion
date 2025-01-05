import ai_diffusion.resources as res
import json


def test_resources_json():
    result = {}
    result["required"] = [m.as_dict() for m in res.required_models]
    result["checkpoints"] = [m.as_dict() for m in res.default_checkpoints]
    result["upscale"] = [m.as_dict() for m in res.upscale_models]
    result["optional"] = [m.as_dict() for m in res.optional_models]
    result["prefetch"] = [m.as_dict() for m in res.prefetch_models]
    result["deprecated"] = [m.as_dict() for m in res.deprecated_models]

    original = res._models_file.read_text()
    assert json.dumps(result, indent=2) == original
