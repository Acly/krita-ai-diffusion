# Stage 1: Base
FROM nvidia/cuda:12.0.1-base-ubuntu22.04 as base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Install Ubuntu packages
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
    nginx \
    net-tools \
    inetutils-ping \
    openssh-server \
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    libxrender1 \
    libxext6 \
    wget \
    curl \
    psmisc \
    rsync \
    vim \
    zip \
    unzip \
    p7zip-full \
    htop \
    pkg-config \
    libtcmalloc-minimal4 \
    apt-transport-https \
    ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Pytorch
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy ComfyUI
COPY ComfyUI /ComfyUI

# Install ComfyUI & custom node dependencies
RUN pip3 install -r /ComfyUI/requirements.txt && \
    pip3 install -r /ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt && \
    pip3 install insightface onnxruntime && \
    pip3 cache purge 

# Copy models
RUN mkdir -p models
COPY downloads/models/ /models/
COPY extra_model_paths.yaml /ComfyUI/

# Install Jupyter
RUN pip3 install -U --no-cache-dir jupyterlab \
    jupyterlab_widgets \
    ipykernel \
    ipywidgets \
    gdown

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# NGINX Proxy
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/502.html /usr/share/nginx/html/502.html
COPY nginx/README.md /usr/share/nginx/html/README.md

# Copy the scripts and create workspace
COPY --chmod=755 pre_start.sh start.sh ./
RUN mkdir -p workspace/logs

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD [ "/start.sh" ]