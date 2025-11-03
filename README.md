# Patient-Adaptive Focused Transmit Beamforming using Cognitive Ultrasound

The repo contains the code for the paper [Patient-Adaptive Focused Transmit Beamforming using Cognitive Ultrasound](https://tue-bmd.github.io/ulsa/). For more information, please refer to the [project page](https://tue-bmd.github.io/ulsa/).

Find the weights of our model on [Huggingface](https://huggingface.co/zeahub/ulsa).

## Setup code

### 1. Settings

```bash
cp .env.example .env  # edit!
cp users.yaml.example users.yaml  # edit!
```

### 2. Dependencies

Install [`zea`](https://github.com/tue-bmd/zea), the cognitive ultrasound toolbox.

```bash
pip install "zea==0.0.7"
```

or use the submodule in this repo:

```bash
git submodule update --init --recursive
pip install -e zea
```

Install other dependencies for this repo:

```bash
KERAS_VER=$(python3 -c "import keras; print(keras.__version__)")
pip install tf2jax==0.3.6 pandas jaxwt dm-pix jax
pip install keras==${KERAS_VER}
```

Alternatively, we have provided a [Dockerfile](./Dockerfile) to build a Docker image with all dependencies installed.

## Dataset

- Download the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/index.html#dataset).

- [_Optionally_] Download the train / validation / test split we used for the [EchoNet-Dynamic dataset](https://huggingface.co/datasets/zeahub/echonet-dynamic).

- Convert the dataset to the polar format:

    ```bash
    python -m zea.data.convert.echonet --source /path/to/echonet-dynamic --target /path/to/echonet-dynamic-polar --splits /path/to/splits
    ```

## Inference

The main file to use for inference is [`active_sampling_temporal.py`](./active_sampling_temporal.py) in combination with a config file.

```bash
python active_sampling_temporal.py --config "configs/echonet_3_frames.yaml"
```

For the 3D model, use [`active_sampling_temporal_3d.py`](./active_sampling_temporal_3d.py).

```bash
python active_sampling_temporal_3d.py --config "configs/elevation_3d.yaml"
```

Additionally, we have created a [Jupyter notebook](./example.ipynb), stripping down the code to the essentials.

## Scripts for paper

In the [`benchmarking_scripts/`](./benchmarking_scripts/) folder, we have provided scripts to reproduce the results from the paper.
These scripts will save data to a folder, which can be visualized using the scripts in the [`plotting/`](./plotting/) folder.
