#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data

wget -O ade20k.zip https://keeper.mpdl.mpg.de/f/80b2fc97ffc3430c98de/?dl=1
unzip ade20k.zip
rm ade20k.zip