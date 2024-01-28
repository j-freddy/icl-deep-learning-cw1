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
# slurm-tune-hparams
# ==============================================================================

for lr in 1e-3 1e-2 1e-1
do
    for wd in 1e-4 1e-3 1e-2
    do
        for batch_size in 32 64 128 256
        do
            model_name="lr${lr}-wd${wd}-batch${batch_size}"

            python -m train --model_save_path ${model_name} --lr $lr --wd $wd --batch_size $batch_size --epochs 20
        done
    done
done
