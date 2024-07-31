#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate
rsync -au --remove-source-files /ComfyUI/* /workspace/ComfyUI

python /scripts/download_models.py ${@} /

cd /workspace/ComfyUI
python main.py --listen --port 3000 &
