from PyQt5.QtCore import Qt

from ai_diffusion.api import (
    WorkflowInput,
    WorkflowKind,
    ControlInput,
    ExtentInput,
    ImageInput,
    TextInput,
)
from ai_diffusion.image import Extent, Image, ImageFileFormat
from ai_diffusion.resources import ControlMode


def test_defaults():
    input = WorkflowInput(WorkflowKind.refine)
    data = input.to_dict()
    assert data == {"kind": "refine", "control": []}
    result = WorkflowInput.from_dict(data)
    assert result == input


def test_serialize():
    input = WorkflowInput(WorkflowKind.generate)
    input.images = ImageInput(ExtentInput(Extent(1, 1), Extent(2, 2), Extent(3, 3), Extent(4, 4)))
    input.images.initial_image = Image.create(Extent(2, 2), Qt.GlobalColor.green)
    input.text = TextInput("prompt")
    input.control = [
        ControlInput(
            ControlMode.line_art, Image.create(Extent(2, 2), Qt.GlobalColor.red), 0.4, (0.1, 0.9)
        ),
        ControlInput(
            ControlMode.depth, Image.create(Extent(4, 2), Qt.GlobalColor.blue), 0.8, (0.2, 0.5)
        ),
    ]

    data = input.to_dict(ImageFileFormat.webp_lossless)
    result = WorkflowInput.from_dict(data)
    assert (
        result.images.initial_image.to_numpy_format()
        == input.images.initial_image.to_numpy_format()
    )
    assert result.control[0].image.to_numpy_format() == input.control[0].image.to_numpy_format()
    assert result.control[1].image.to_numpy_format() == input.control[1].image.to_numpy_format()
    assert result == input
