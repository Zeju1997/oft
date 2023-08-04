import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from oft_utils.unet_2d_condition import UNet2DConditionModel

model_base = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet = UNet2DConditionModel.from_pretrained(
    model_base, subfolder="unet", torch_dtype=torch.float32
)

oft_model_path = "./sddata/finetune/oft/sketch/checkpoint-10"
pipe.unet.load_attn_procs(oft_model_path)
pipe.to("cuda")

image = pipe(
    "giraffe is eating leaves from the tree", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}
).images[0]

# image = pipe("A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("sketch.png")