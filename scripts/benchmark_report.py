import sys
from pathlib import Path

from PIL import Image

prefixes = [
    "benchmark_inpaint_apple-tree",
    "benchmark_inpaint_bruges",
    "benchmark_inpaint_girl-cornfield",
    "benchmark_inpaint_jungle",
    "benchmark_inpaint_nature",
    "benchmark_inpaint_park",
    "benchmark_inpaint_street",
    "benchmark_inpaint_superman",
    "benchmark_inpaint_tori",
]

names = [
    "Apple tree - illustration - outpaint",
    "Canal - photo - inpaint",
    "Cornfield - anime - outpaint",
    "Jungle - painting - inpaint",
    "Nature - photo - inpaint",
    "Park - photo - inpaint",
    "Street - photo - inpaint",
    "Superman - illustration - inpaint",
    "Torii - photo - inpaint",
]

suffixes = [
    "_sdxl_noprompt_4213_local.png",
    "_sdxl_noprompt_897281_local.png",
    "_sdxl_prompt_4213_local.png",
    "_sdxl_prompt_897281_local.png",
]


def main(folder):
    path = Path(folder)
    out = path / "compressed"
    out.mkdir(exist_ok=True)
    for prefix in prefixes:
        images = []
        for suffix in suffixes:
            p = path / (prefix + suffix)
            if p.exists():
                img = Image.open(p)
                img = img.resize((480, img.height * 480 // img.width), Image.Resampling.LANCZOS)
                images.append(img)
        strip = Image.new("RGB", (480 * len(images), images[0].height))
        for i, img in enumerate(images):
            strip.paste(img, (480 * i, 0))
        strip.save(out / (prefix + ".jpg"), "JPEG", quality=90)

    with open(path / "results.md", "w") as f:
        f.write("# Results\n")
        for name, prefix in zip(names, prefixes):
            f.write(f"\n## {name}\n")
            f.write(f"![{name}](compressed/{prefix}_input.jpg)\n\n")
            f.write("Results\n")
            f.write(f"![{name}](compressed/{prefix}.jpg)\n")

            img = Image.open(prefix.replace("benchmark_inpaint_", "") + ".png")
            img = img.resize((320, img.height * 320 // img.width), Image.Resampling.LANCZOS)
            img.save(out / (prefix + "_input.jpg"), "JPEG", quality=90)


if __name__ == "__main__":
    main(sys.argv[1])
