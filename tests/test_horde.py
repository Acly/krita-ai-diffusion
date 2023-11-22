from ai_diffusion import network
from ai_diffusion.image import Image
import asyncio

url = "https://aihorde.net"
headers = {"apikey": "23Ta45w2IH3aecE2S2ThhA"}


def test_simple(qtapp):
    async def main():
        net = network.RequestManager()
        payload = {
            "prompt": "dancing cactus",
            "seed": "1234",
            "post_processing": [],
            "models": ["Realistic Vision"],
        }
        try:
            job = await net.post(f"{url}/api/v2/generate/async", payload, headers)
            print(job)
            id = job["id"]
        except network.NetworkError as e:
            print(e)
            return

        while True:
            status = await net.get(f"{url}/api/v2/generate/check/{id}")
            print(status)
            if status["done"] or status["faulted"]:
                break

            await asyncio.sleep(2)

        status = await net.get(f"{url}/api/v2/generate/status/{id}")
        print(status)

        img_link = status["generations"][0]["img"]
        print("download image", img_link)
        img_data = await net.get(img_link)
        img = Image.from_bytes(img_data, "WEBP")
        img.save("test.webp")

    qtapp.run(main())
