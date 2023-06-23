#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data

wget -O deepfashion.zip "https://keeper.mpdl.mpg.de/f/e3d1e4294b3f49a5aff0/?dl=1"
unzip deepfashion.zip
rm deepfashion.zip