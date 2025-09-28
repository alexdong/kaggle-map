FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    git-lfs \
    sudo \
    curl \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && python3 -m pip install --upgrade pip

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir /usr/local/bin --force

RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
 && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

ENV UV_PROJECT_ENVIRONMENT=workspace
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

WORKDIR /workspaces/kaggle-map

RUN mkdir -p /workspaces/kaggle-map \
 && chown -R ${USERNAME}:${USERNAME} /workspaces

USER ${USERNAME}
