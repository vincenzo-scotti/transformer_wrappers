# Transformer Wrappers

Codebase for the project ...
TODO add project description and link to Overleaf/paper

## Repository structure

This repository is organised into four main directories:

```
|- experiments/
  |- ...
|- notebooks/
  |- ...
|- docker/
  |- ...
|- resources/
  |- configs/
    |- ...
  |- data/
    |- ...
  |- models/
    |- ...
|- src/
  |- script/
    |- ...
  |- transformer_wrappers
    |- ...
```

For further details, refer to the `README.md` within each directory.

## Environment setup

> [!NOTE]  
> These instructions were written for Ubuntu 22.04 using NVIDIA drivers 535 and CUDA 11.8; make sure to apply the appropriate changes depending on your system.

The environment can be set up either via Docker (suggested) or by creating it manually.
To run the code, it's necessary to install the NVIDIA drivers. Refer to [this script](https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba?permalink_comment_id=4715433) to help you in the process.

### Docker 

To use the Docker containers, it's necessary to install and configure Docker ([installation guide](https://docs.docker.com/engine/install/ubuntu/)). Additionally, make sure to add the current use to the Docker group ([post installation guide](https://docs.docker.com/engine/install/linux-postinstall/)).
To use the GPUs in the container, it's necessary to install Nvidia Docker as well ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

Build the image with the project environment.

```bash
docker build . -f ./docker/trwrap/Dockerfile -t trwrap
```

Start the container and connect it to be sure everything is working correctly.

```bash
docker \
  run \
  -v /path/to/transformer_wrappers/resources:/app/transformer_wrappers/resources \
  -v /path/to/transformer_wrappers/experiments:/app/transformer_wrappers/experiments \
  -v /path/to/transformer_wrappers/notebooks:/app/transformer_wrappers/notebooks \
  -v /path/to/transformer_wrappers/src:/app/transformer_wrappers/src \
  --gpus all \
  --network="host" \
  --name trwrap \
  -it trwrap \
  /bin/bash
```

### Manual

To install all the required packages within an Anaconda environment, run the following commands:

```bash
# Create anaconda environment 
conda create -n trwrap python=3.12
# Activate anaconda environment
conda activate trwrap
# Install required packages
conda install cuda -c nvidia
pip install -r requirements.txt
```

To add the source code directory(ies) to the Python path, you can add this line to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/transformer_wrappers/src
```

To run the unit tests you will need to add the Hugging-Face authentication token to your environment variables:

```bash
export HUGGING_FACE_TOKEN=...
```

## References

If you are willing to use our code or our models, please cite us with the following reference(s):

```bibtex
...
```


## Acknowledgements

- Nicolò Brunello ([nicolo.brunello@polimi.it](mailto:nicolo.brunello@polimi.it))
- Mark James Carman: ([mark.carman@polimi.it](mailto:mark.carman@.polimi.it))
- Davide Rigamonti: ([davide2.rigamonti@mail.polimi.it](mailto:davide2.rigamonti@mail.polimi.it))
- Vincenzo Scotti: ([vincenzo.scotti@polimi.it](mailto:vincenzo.scotti@polimi.it))
