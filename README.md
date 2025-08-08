# Patient-Adaptive Focused Transmit Beamforming using Cognitive Ultrasound

[Huggingface](https://huggingface.co/zeahub/ulsa)

## Setup

```bash
git submodule update --init --recursive
pip install -e zea
KERAS_VER=$(python3 -c "import keras; print(keras.__version__)")
pip install tf2jax==0.3.6 pandas jaxwt dm-pix jax
pip install keras==${KERAS_VER}
cp .env.example .env
touch users.yaml # edit!
```
