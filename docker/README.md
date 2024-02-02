# Docker

This directory is used to host the [Docker](https://www.docker.com) files to build the container images.

> [!NOTE]  
> Container files have some paths hardcoded inside, make sure to update them accordingly when launching the code.

## Container

There is a single container described by the Dockerfile in `trwrap/`.

## Setup

Here follows the setup instruction to use Docker and NVIDIA Docker to run the containers.

### NVIDIA drivers

The following instructions are to install drivers version 535 for CUDA 11.8 on Ubuntu 22.04 adapted from [this script](https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba?permalink_comment_id=4715433), make sure to do all the changes for your system.

```bash
# To verify your gpu is cuda enable check
lspci | grep -i nvidia

# If you have previous installation remove it first.
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

# System update
sudo apt-get update
sudo apt-get upgrade

# Install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# First get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install nvidia driver with dependencies
sudo apt install libnvidia-common-535
sudo apt install libnvidia-gl-535
sudo apt install nvidia-driver-535

sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

# Need to sudo apt-get upgrade or the next step wont work
sudo apt upgrade
```

### Docker

Install and configure ([installation guide](https://docs.docker.com/engine/install/ubuntu/)).

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo service docker start
```

Additionally, make sure to add the current use to the Docker group ([post installation guide](https://docs.docker.com/engine/install/linux-postinstall/)).

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### NVIDIA Docker

Install and configure ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& \
  sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd

sudo nvidia-ctk runtime configure --runtime=crio
sudo systemctl restart crio
```
