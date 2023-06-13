from ai_tools import workflow, Mask, Bounds, Extent, Image, Progress
from pathlib import Path

test_dir = Path(__file__).parent
image_dir = test_dir / 'images'
result_dir = test_dir / '.results'

def print_progress(value):
    print(f'Progress: {value * 100:.1f}%')

def test_inpaint(qtapp):
    image = Image.load(image_dir / 'beach_768x512.png')
    mask = Mask.rectangle(Bounds(50, 100, 320, 200), feather=10)
    async def main():
        result = await workflow.inpaint(image, mask, 'ship', Progress(print_progress))
        result.save(result_dir / 'test_inpaint.png')
    qtapp.run(main())
    
def test_inpaint_upscale(qtapp):
    image = Image.load(image_dir / 'beach_1536x1024.png')
    mask = Mask.rectangle(Bounds(600, 200, 768, 512), feather=10)
    async def main():
        result = await workflow.inpaint(image, mask, 'ship', Progress(print_progress))
        result.save(result_dir / 'test_inpaint_upscale.png')
    qtapp.run(main())
    