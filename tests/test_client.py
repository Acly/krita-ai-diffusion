import pytest

from ai_diffusion import Client, ClientEvent, ComfyWorkflow, NetworkError, Image, Extent
from ai_diffusion.client import parse_url

default_checkpoint = "realisticVisionV51_v51VAE.safetensors"


def make_default_workflow(steps=20):
    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint(default_checkpoint)
    positive = w.clip_text_encode(clip, "a photo of a cat")
    negative = w.clip_text_encode(clip, "a photo of a dog")
    latent_image = w.empty_latent_image(512, 512)
    latent_result = w.ksampler(model, positive, negative, latent_image, steps=steps)
    result_image = w.vae_decode(vae, latent_result)
    w.send_image(result_image)
    return w


def make_trivial_workflow():
    img = Image.create(Extent(16, 16))
    w = ComfyWorkflow()
    w.send_image(w.load_image(img))
    return w


def test_connect_bad_url(qtapp):
    async def main():
        with pytest.raises(NetworkError):
            await Client.connect("bad_url")

    qtapp.run(main())


@pytest.mark.parametrize("cancel_point", ["after_enqueue", "after_start", "after_sampling"])
def test_cancel(qtapp, cancel_point):
    async def main():
        client = await Client.connect()
        job_id = await client.enqueue(make_default_workflow(steps=200))
        assert client.queued_count == 1
        if cancel_point == "after_enqueue":
            await client.interrupt()

        interrupted = False
        async for msg in client.listen():
            assert msg.event is not ClientEvent.finished
            assert msg.job_id == job_id
            if not interrupted:
                if cancel_point == "after_start":
                    await client.interrupt()
                    interrupted = True
                if cancel_point == "after_sampling" and msg.progress > 0.1:
                    await client.interrupt()
                    interrupted = True
            if msg.event is ClientEvent.interrupted:
                assert msg.job_id == job_id
                break
            else:
                assert client.is_executing
        assert client.is_executing == False and client.queued_count == 0

        job_id = await client.enqueue(make_trivial_workflow())
        assert client.queued_count == 1
        async for msg in client.listen():
            assert msg.event is not ClientEvent.interrupted
            assert msg.job_id == job_id
            if msg.event is ClientEvent.finished:
                assert msg.images[0].extent == Extent(16, 16)
                break

        assert client.is_executing == False and client.queued_count == 0

    qtapp.run(main())


def test_disconnect(qtapp):
    async def main():
        client = await Client.connect()
        await client.disconnect()
        assert client.is_executing == False and client.queued_count == 0

    qtapp.run(main())


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8000", ("http://localhost:8000", "ws://localhost:8000")),
        ("http://localhost:8000/", ("http://localhost:8000", "ws://localhost:8000")),
        ("http://localhost:8000/foo", ("http://localhost:8000/foo", "ws://localhost:8000/foo")),
        ("http://127.0.0.1:1234", ("http://127.0.0.1:1234", "ws://127.0.0.1:1234")),
        ("localhost:8000", ("http://localhost:8000", "ws://localhost:8000")),
        ("https://localhost:8000", ("https://localhost:8000", "wss://localhost:8000")),
    ],
)
def test_parse_url(url, expected):
    assert parse_url(url) == expected
