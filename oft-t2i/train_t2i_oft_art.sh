export HF_HOME='/tmp'

# eps=6e-5
# r=4

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate oft

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./sddata/finetune/oft/wikiart"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="fusing/wikiart_captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_oft.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=10000 \
  --validation_prompt="giraffe is eating leaves from the tree" \
  --seed=1337 \
  --eps=6e-5 \
  --r=4 \
  # --coft \

  # 15000
  # 500