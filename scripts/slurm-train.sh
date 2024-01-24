#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL   # required to send email notifcations
#SBATCH --mail-user=ffj20 # required to send email notifcations

. /vol/cuda/12.1.0/setup.sh
export LD_LIBRARY_PATH=/vol/bitbucket/${USER}/deep-learning-cw1/venv/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}
TERM=vt100                # or TERM=xterm
/usr/bin/nvidia-smi
uptime

cd deep-learning-cw1
export PATH=/vol/bitbucket/${USER}/deep-learning-cw1/venv/bin/:$PATH
source activate

# ==============================================================================
# slurm-train
# ==============================================================================

python -m train --model_save_path model_augmented.pt
