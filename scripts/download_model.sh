#!/bin/bash

echo -e "\nDownloading stable diffusion ckpt..."

cd models
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt

echo -e "\nDone."
