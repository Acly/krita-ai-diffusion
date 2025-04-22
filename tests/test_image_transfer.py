from base64 import b64decode
from PIL import Image
from datetime import datetime
import os
import pytest

from ai_diffusion.image import ImageCollection, Image as ImageWrapper
from ai_diffusion.cloud_client import CloudClient
from .config import root_dir, test_dir

if (root_dir / "service" / "pod" / "lib").exists():
    import dotenv
    import requests

    dotenv.load_dotenv(root_dir / "service" / "web" / ".env.local")
    from service.pod.lib import image_transfer
    from service.pod.lib import log

    max_b64_size_config = {
        "transfer": 100_000,  # use R2 for images > 100kb -> will use R2
        "b64": 5_000_000,  # use R2 for images > 5mb -> will use b64
    }

    @pytest.mark.parametrize("format", ["webp", "png"])
    @pytest.mark.parametrize("mode", ["b64", "transfer"])
    def test_send(qtapp, format: str, mode: str):
        images = [
            Image.open(test_dir / "images" / f).convert("RGBA")
            for f in ("cat.webp", "pegonia.webp")
        ]
        max_b64_size = max_b64_size_config[mode]

        async def main():
            logger = log.Log("test")
            metrics = log.Metrics("test", datetime.now())
            transfer = await image_transfer.send_images(
                images, metrics, logger, max_inline_size=max_b64_size, format=format
            )
            assert len(transfer["offsets"]) == 2

            if mode == "transfer":
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

        qtapp.run(main())

    @pytest.mark.parametrize("mode", ["b64", "transfer"])
    def test_receive(qtapp, mode: str):
        max_b64_size = max_b64_size_config[mode]
        images = [ImageWrapper.load(test_dir / "images" / f) for f in ("cat.webp", "pegonia.webp")]
        bytes, offsets = ImageCollection(images).to_bytes()
        input = {"image_data": {"bytes": bytes, "offsets": offsets}}

        async def main():
            url = os.environ["TEST_SERVICE_URL"]
            token = os.environ.get("TEST_SERVICE_TOKEN", "")
            client = await CloudClient.connect(url, token)
            await client.send_images(input, max_inline_size=max_b64_size)

            if mode == "transfer":
                assert "s3_object" in input["image_data"]
            else:
                assert "base64" in input["image_data"]
            await image_transfer.receive_images(input)

            image_data = input["image_data"]
            blob, offsets = image_data["bytes"], image_data["offsets"]
            result = ImageCollection.from_bytes(blob, offsets)
            assert len(result) == 2
            for result, expected in zip(result, images):
                assert ImageWrapper.compare(result, expected) < 0.01

        qtapp.run(main())
