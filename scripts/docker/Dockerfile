FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS base

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
    bash \
    curl \
    vim \
    htop \
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

# Install Python virtual environment
ENV XDG_BIN_HOME="/usr/bin" \
    PATH="/venv/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv /venv --python python3.12

# Install Python dependencies
COPY ./requirements.txt /requirements.txt
RUN uv pip install --no-cache -r requirements.txt
RUN uv pip install --no-cache https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.8-cp312-cp312-linux_x86_64.whl

# Install ComfyUI
COPY ./ComfyUI /ComfyUI/
COPY ./extra_model_paths.yaml /ComfyUI/

# Prepare scripts and folders for models (download is eventually done at container startup via pre_start.sh)
COPY ./krita-ai-diffusion/ /krita-ai-diffusion/
RUN python3 /krita-ai-diffusion/scripts/download_models.py --dry-run /workspace
RUN ln -s /workspace/models /models


FROM base

# Install additional dependencies (convenience for manual interaction)
RUN apt update && \
    apt install -y --no-install-recommends \
    nginx \
    net-tools \
    inetutils-ping \
    openssh-server \
    wget \
    psmisc \
    rsync \
    zip \
    unzip \
    p7zip-full \
    apt-transport-https && \    
    apt clean

# Install Jupyter
RUN uv pip install --no-cache jupyterlab \
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
