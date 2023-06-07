import os 

# Images with width/height larger will be downscaled before inpaint
max_inpaint_resolution = 768

# Folder where intermediate images are stored for debug purposes (default: None)
debug_image_folder = os.environ.get('KRITA_AI_TOOLS_DEBUG_IMAGE')
