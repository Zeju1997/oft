# export HF_HOME='/tmp'
. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate oft
parms=$1
echo $parms

python test_text_to_image.py --img_ID=$parms