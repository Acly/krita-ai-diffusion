# Stage 1: Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

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
        python3.10-venv \
        python3-pip \
        python3-tk \
        python3-dev \
        bash \
        dos2unix \
        git \
        git-lfs \
        ncdu \
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
        plocate \
        libcairo2-dev \
        libgoogle-perftools4 \
        libtcmalloc-minimal4 \
        apt-transport-https \
        ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install -r /ComfyUI/requirements.txt && \
    pip3 cache purge && \
    echo v1.13.0 > .krita-ai-diffusion.version

# Install ComfyUI Custom Nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux /ComfyUI/custom_nodes/comfyui_controlnet_aux && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus /ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale /ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale --recursive && \
    git clone https://github.com/Acly/comfyui-tooling-nodes /ComfyUI/custom_nodes/comfyui-tooling-nodes && \
    pip3 install -r /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt && \    
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

# Install runpodctl
RUN wget https://github.com/runpod/runpodctl/releases/download/v1.10.0/runpodctl-linux-amd -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

# Install croc
RUN curl https://getcroc.schollz.com | bash

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