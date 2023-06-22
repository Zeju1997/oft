export HF_HOME='/tmp'

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate nerf

python eval.py
# python eval_ablation.py