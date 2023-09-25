#!/bin/bash

export PYTHONUNBUFFERED=1
rsync -au --remove-source-files /ComfyUI/* /workspace/ComfyUI

cd /workspace/ComfyUI
python main.py --listen --port 3000 &
