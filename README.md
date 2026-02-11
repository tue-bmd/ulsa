# Patient-Adaptive Echocardiography using Cognitive Ultrasound

The repo contains the code for the paper [Patient-Adaptive Echocardiography using Cognitive Ultrasound](https://tue-bmd.github.io/ulsa/). For more information, please refer to the [project page](https://tue-bmd.github.io/ulsa/).

Find the weights of our model on [Huggingface](https://huggingface.co/zeahub/ulsa).

![measurements_reconstruction_0X10A5FC19152B50A5](https://github.com/user-attachments/assets/c5ed6f17-fdc6-4b4a-840b-b92095439f28)

## Setup code

### 1. Settings

```bash
cp .env.example .env  # edit!
cp users.yaml.example users.yaml  # edit!
```

### 2. Dependencies

Install this repository in editable mode:

```bash
pip install -e .
```

Install [`zea`](https://github.com/tue-bmd/zea), the cognitive ultrasound toolbox, preferably
through the the submodule in this repo:

```bash
git submodule update --init --recursive
pip install -e zea
```

Install other dependencies for this repo:

```bash
KERAS_VER=$(python3 -c "import keras; print(keras.__version__)")
pip install tf2jax==0.3.6 pandas jaxwt jax
pip install keras==${KERAS_VER}
```

Alternatively, we have provided a [Dockerfile](./Dockerfile) to build a Docker image with all dependencies installed.

## Dataset

- Download the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/index.html#dataset).

- [_Optionally_] Download the train / validation / test split we used for the [EchoNet-Dynamic dataset](https://huggingface.co/datasets/zeahub/echonet-dynamic).

- Convert the dataset to the polar format:

    ```bash
    python -m zea.data.convert echonet /path/to/echonet-dynamic /path/to/echonet-dynamic-polar --split_path /path/to/split.yaml
    ```

## Training

To train the video diffusion model, use the `models/train_diffusion.py` script. The [time conditional U-Net architecture](https://github.com/tue-bmd/zea/blob/8613512de47d64ae72e5fe03f30c47dfd2b12f46/zea/models/unet.py#L157) implemented by [`zea`](https://zea.readthedocs.io/en/latest/) is used for the denoiser. You can modify architectural and training hyperparameters in the config `configs/training/echonet_diffusion_3_frames.yaml`.

```bash
python models/train_diffusion.py
```


## Inference

The main file to use for inference is [`ulsa/active_sampling_temporal.py`](./ulsa/active_sampling_temporal.py) in combination with a config file.

```bash
python ulsa/active_sampling_temporal.py --config "configs/echonet_3_frames.yaml"
```

For the 3D model, use [`ulsa/active_sampling_temporal_3d.py`](./ulsa/active_sampling_temporal_3d.py).

```bash
python ulsa/active_sampling_temporal_3d.py --config "configs/elevation_3d.yaml"
```

For educational purposes, we have also created a simplified version of our algorithm [in this notebook](https://zea.readthedocs.io/en/latest/notebooks/agent/agent_example.html).

## Scripts for paper

In the [`benchmarking_scripts/`](./benchmarking_scripts/) folder, we have provided scripts to reproduce the results from the paper.
These scripts will save data to a folder, which can be visualized using the scripts in the [`plotting/`](./plotting/) folder.
