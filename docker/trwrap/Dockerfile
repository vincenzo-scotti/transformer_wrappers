FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL description="Transformer Wrappers project environment (docs: https://github.com/vincenzo-scotti/transformer_wrappers)."
# Install required tools
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
# Create working directory
RUN mkdir /app
RUN mkdir /app/transformer_wrappers
# Declare volumes
VOLUME /app/transformer_wrappers/resources
VOLUME /app/transformer_wrappers/experiments
VOLUME /app/transformer_wrappers/notebooks
VOLUME /app/transformer_wrappers/src
# Set current working directory
WORKDIR /app/transformer_wrappers
# Copy requirement files
COPY requirements.txt .requirements.txt
# Install required packages
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r .requirements.txt
# Update Python path
ENV PYTHONPATH "${PYTHONPATH}:/app/transformer_wrappers/src"
