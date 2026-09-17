#!/bin/bash
#SBATCH -n 4
#SBATCH -w gpu01
#SBATCH --gres=gpu:1

# 强制UTF-8环境 ====
export LANG=en_US.UTF-8  # 或 zh_CN.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1  # Python 3.7+ 专用选项

# 原有conda初始化部分 ====
__conda_setup="$(CONDA_REPORT_ERRORS=false '/gpu01/miniconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/gpu01/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/gpu01/miniconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/gpu01/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# 环境
conda activate old_pytorch
date

cd $SLURM_SUBMIT_DIR
source /opt/rh/devtoolset-11/enable
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT


# 文件
python train.py 