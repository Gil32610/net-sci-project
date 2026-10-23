FROM python:3.11-bookworm

ARG USERNAME=vscode
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $GID $USERNAME \
    && useradd --uid $UID --gid $GID -m $USERNAME

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
USER $USERNAME