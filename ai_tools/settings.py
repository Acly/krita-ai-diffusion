import os

# Images with width and height smaller will be upscaled before diffusion
min_image_size = 512

# Images with width or height larger will be downscaled before diffusion
max_image_size = 768

# Number of low-res images which are generated at once (must all fit into VRAM)
batch_size = 2

# Folder where intermediate images are stored for debug purposes (default: None)
debug_image_folder = os.environ.get("KRITA_AI_TOOLS_DEBUG_IMAGE")
