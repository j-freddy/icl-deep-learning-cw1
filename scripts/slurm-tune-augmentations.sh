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

for crop_scale in 0.1 0.2 0.3
do
    for color_bcs in 0.4 0.5 0.6
    do
        for color_hue in 0.05 0.1 0.15
        do
            for jitter_p in 0.6 0.75 0.9
            do
                model_name="model_cropscale${crop_scale}_bcs${color_bcs}_hue${color_hue}_jitterp${jitter_p}.pt"

                python -m train --model_save_path $model_name --aug_crop_scale $crop_scale --aug_color_bcs $color_bcs --aug_color_hue $color_hue --aug_jitter_p $jitter_p
            done
        done
    done
done
