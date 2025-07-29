# This is run in the container prior to the python script.
export PYTHONPATH=$PYTHONPATH:/ulsa
export SNELLIUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MACHINE=snellius

export WANDB_DIR=/projects/0/prjs0966/$USER/wandb
export WANDB_CONFIG_DIR=/projects/0/prjs0966/$USER/wandb
export WANDB_CACHE_DIR=/projects/0/prjs0966/$USER/wandb

echo "WANDB_DIR: $WANDB_DIR"
echo "WANDB_CONFIG_DIR: $WANDB_CONFIG_DIR"
echo "WANDB_CACHE_DIR: $WANDB_CACHE_DIR"

mkdir -p $WANDB_DIR
mkdir -p $WANDB_CONFIG_DIR
mkdir -p $WANDB_CACHE_DIR

echo "Running $@"
"$@"