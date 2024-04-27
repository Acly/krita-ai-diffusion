from base64 import b64decode
from PIL import Image
import pytest

from ai_diffusion.image import ImageCollection, Image as ImageWrapper
from .config import root_dir, test_dir

if (root_dir / "service" / "pod" / "lib").exists():
    import dotenv
    import requests

    dotenv.load_dotenv(root_dir / "service" / "web" / ".env.local")
    from service.pod.lib import image_transfer

    @pytest.mark.parametrize("mode", ["b64", "s3"])
    def test_send(mode: str):
        images = [
            Image.open(test_dir / "images" / f).convert("RGBA")
            for f in ("cat.webp", "pegonia.webp")
        ]
        max_b64_size = {
            "s3": 100_000,  # use s3 for images > 100kb -> will use s3
            "b64": 5_000_000,  # use s3 for images > 5mb -> will use b64
        }[mode]
        transfer = image_transfer.send_images(images, max_inline_size=max_b64_size)
        assert len(transfer["offsets"]) == 2

        if mode == "s3":
            url = transfer.get("url")
            assert url and "interstice-transfer-1" in url
            response = requests.get(url)
            assert response.status_code == 200
            result_bytes = response.content
        else:
            b64data = transfer.get("base64")
            assert isinstance(b64data, str)
            result_bytes = b64decode(b64data.encode("utf-8"))

        results = ImageCollection.from_bytes(result_bytes, transfer["offsets"])
        for result, expected in zip(results, images):
            assert result.to_numpy_format() == ImageWrapper.from_pil(expected).to_numpy_format()
