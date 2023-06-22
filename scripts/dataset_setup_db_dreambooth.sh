#!/bin/bash

cd ..

if [ ! -d "data" ]; then
  mkdir data
fi
cd data

wget https://owncloud.tuebingen.mpg.de/index.php/s/pzciprZpTPXD8Lq/download -O vggface2.tar
tar xvf vggface2.tar
rm vggface2.tar