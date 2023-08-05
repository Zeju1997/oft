export HF_HOME='/tmp'

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="./data/COCO"
export OUTPUT_DIR="./sddata/finetune/sd/coco"

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=200000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=5000 \
  --validation_prompt="A beautiful woman taking a picture with her smart phone." \
  --seed=1337 \

  # 150000