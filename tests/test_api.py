from PyQt5.QtCore import Qt

from ai_diffusion.api import (
    ConditioningInput,
    ControlInput,
    ExtentInput,
    ImageInput,
    RegionInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.image import Bounds, Extent, Image, ImageFileFormat
from ai_diffusion.resources import ControlMode
from ai_diffusion.util import ensure


def test_defaults():
    input = WorkflowInput(WorkflowKind.refine)
    data = input.to_dict()
    assert data == {"kind": "refine"}
    result = WorkflowInput.from_dict(data)
    assert result == input


def _ensure_cmp(img: Image | None):
    return ensure(img).to_numpy_format()


def test_serialize():
    input = WorkflowInput(WorkflowKind.generate)
    input.images = ImageInput(ExtentInput(Extent(1, 1), Extent(2, 2), Extent(3, 3), Extent(4, 4)))
    input.images.initial_image = Image.create(Extent(2, 2), Qt.GlobalColor.green)
    input.conditioning = ConditioningInput(
        "prompt",
        control=[
            ControlInput(
                ControlMode.line_art,
                Image.create(Extent(2, 2), Qt.GlobalColor.red),
                0.4,
                (0.1, 0.9),
            ),
            ControlInput(
                ControlMode.depth, Image.create(Extent(4, 2), Qt.GlobalColor.blue), 0.8, (0.2, 0.5)
            ),
            ControlInput(ControlMode.blur, None, 0.5),
        ],
    )

    data = input.to_dict(ImageFileFormat.webp_lossless)
    result = WorkflowInput.from_dict(data)
    assert result.images is not None and result.images.initial_image is not None
    assert (
        result.images.initial_image.to_numpy_format()
        == input.images.initial_image.to_numpy_format()
    )
    input_control = ensure(input.conditioning).control
    result_control = ensure(result.conditioning).control
    assert _ensure_cmp(result_control[0].image) == _ensure_cmp(input_control[0].image)
    assert _ensure_cmp(result_control[1].image) == _ensure_cmp(input_control[1].image)
    assert result_control[2].image is None
    assert result == input


def test_deserialize_list_default():
    input = WorkflowInput(WorkflowKind.generate)
    input.images = ImageInput(ExtentInput(Extent(1, 1), Extent(2, 2), Extent(3, 3), Extent(4, 4)))
    input.conditioning = ConditioningInput(
        "prompt",
        regions=[
            RegionInput(
                Image.create(Extent(2, 2), Qt.GlobalColor.red), Bounds(0, 0, 2, 2), "positive", []
            )
        ],
    )

    data = input.to_dict()
    del data["conditioning"]["regions"][0]["loras"]
    result = WorkflowInput.from_dict(data)
    assert result.conditioning is not None
    assert len(result.conditioning.regions[0].loras) == 0
