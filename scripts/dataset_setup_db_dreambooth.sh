#!/bin/bash

echo -e "\nDownloading dreambooth dataset..."
git clone https://github.com/google/dreambooth.git
mv dreambooth/dataset/* data/dreambooth
rm -rf dreambooth