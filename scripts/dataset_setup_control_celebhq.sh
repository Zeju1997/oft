#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data

wget -O celechq-text.zip https://keeper.mpdl.mpg.de/f/72c34a6017cb40b896e9/?dl=1
unzip celechq-text.zip
rm celechq-text.zip