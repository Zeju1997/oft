export HF_HOME='/tmp'

# eps=6e-5
# r=4

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./sddata/finetune/oft/coco_8e-05"
export HUB_MODEL_ID="pokemon-lora"
export TRAIN_DIR="./data/COCO"

accelerate launch --mixed_precision="fp16"  train_text_to_image_oft.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$TRAIN_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=200000 \
  --learning_rate=8e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=10000 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337 \
  --eps=6e-5 \
  --r=4 \
  # --coft \

  # 15000
  # 500
  # 1e-04