#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100

# before running
: '
git checkout {your-branch}
git pull
git submodule update
cat users.yaml # check if users.yaml is okay
cat .env # check if .env is okay
'
# Check that your python script is set up to handle the --num_shards and --shard_index arguments.
# run this file with: `sbatch --time=00:30:00 --array=0-99 launch/snellius_sharded.sh ...`

# Doc: https://apptainer.org/docs/user/main/cli/apptainer_exec.html
# not binding home to avoid cache conflicts (hugginface, usbmd, etc...!)
srun apptainer exec --env-file .env \
                    --env ZEA_CACHE_DIR="$TMPDIR/$SLURM_ARRAY_TASK_ID" \
                    --env HF_HOME="$TMPDIR/$SLURM_ARRAY_TASK_ID/huggingface_home" \
                    --env MPLCONFIGDIR="$TMPDIR/$SLURM_ARRAY_TASK_ID/matplotlib" \
                    --no-home \
                    --cwd /ulsa/ \
                    --bind ~/ulsa/:/ulsa/ \
                    --bind /projects/0/prjs0966/:/projects/0/prjs0966/ \
                    --nv /projects/0/prjs0966/ulsa.sif \
                    /bin/bash launch/snellius_helper.sh "$@" --shard_index $SLURM_ARRAY_TASK_ID
