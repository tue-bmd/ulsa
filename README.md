# Patient-Adaptive Focused Transmit Beamforming using Cognitive Ultrasound

The repo contains the code for the paper [Patient-Adaptive Focused Transmit Beamforming using Cognitive Ultrasound](https://tue-bmd.github.io/ulsa/). For more information, please refer to the [project page](https://tue-bmd.github.io/ulsa/).

Find the weights of our model on [Huggingface](https://huggingface.co/zeahub/ulsa).

## Setup code

```bash
git submodule update --init --recursive
pip install -e zea
KERAS_VER=$(python3 -c "import keras; print(keras.__version__)")
pip install tf2jax==0.3.6 pandas jaxwt dm-pix jax
pip install keras==${KERAS_VER}
cp .env.example .env
touch users.yaml # edit!
```

## Dataset

- Download the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/index.html#dataset).

- [_Optionally_] Download the train / validation / test split we used for the [EchoNet-Dynamic dataset](https://huggingface.co/datasets/zeahub/echonet-dynamic).

- Convert the dataset to the polar format:

    ```bash
    python -m zea.data.convert.echonet --source /path/to/echonet-dynamic --target /path/to/echonet-dynamic-polar --splits /path/to/splits
    ```
