FROM nvidia/cuda:12.0.1-base-ubuntu22.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Minimal system dependencies
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    python3.11-venv \
    python3.11-dev \
    bash \
    git \
    git-lfs \
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    libxrender1 \
    libxext6 \
    pkg-config \
    libtcmalloc-minimal4 \
    ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Python virtual environment
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# PyTorch and ONNX Runtime
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio onnxruntime

# ComfyUI and custom nodes
COPY ./ComfyUI /ComfyUI/
RUN python3 -m pip install -r /ComfyUI/requirements.txt && \
    python3 -m pip install -r /ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt && \
    python3 -m pip install -r /ComfyUI/custom_nodes/comfyui-tooling-nodes/requirements.txt && \
    python3 -m pip install insightface && \
    python3 -m pip cache purge

# Prepare scripts and folders for models (download is eventually done at container startup via pre_start.sh)
COPY ./krita-ai-diffusion/ /krita-ai-diffusion/
COPY ./extra_model_paths.yaml /ComfyUI/
RUN python3 -m pip install --no-cache-dir aiohttp tqdm && \
    python3 /krita-ai-diffusion/scripts/download_models.py --dry-run /workspace
RUN ln -s /workspace/models /models


FROM base

# Install additional dependencies (convenience for manual interaction)
RUN apt install -y --no-install-recommends \
    nginx \
    net-tools \
    inetutils-ping \
    openssh-server \
    wget \
    curl \
    psmisc \
    rsync \
    vim \
    zip \
    unzip \
    p7zip-full \
    htop \
    apt-transport-https && \    
    apt clean

# ComfyUI Manager dependencies
RUN python3 -m pip install --no-cache-dir \
    -r /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Install Jupyter
RUN python3 -m pip install -U --no-cache-dir jupyterlab \
    jupyterlab_widgets \
    ipykernel \
    ipywidgets \
    gdown

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# NGINX Proxy
COPY ./nginx/nginx.conf /etc/nginx/nginx.conf
COPY ./nginx/502.html /usr/share/nginx/html/502.html
COPY ./nginx/README.md /usr/share/nginx/html/README.md

# Copy the scripts and create workspace
RUN mkdir -p workspace/logs
COPY --chmod=755 pre_start.sh start.sh ./

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD ["/start.sh", "--recommended"]
