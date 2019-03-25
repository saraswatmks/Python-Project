#! /bin/bash

echo "installing miniconda"
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/miniconda
rm ~/miniconda.sh

echo "configure path to run conda"
export PATH=/home/hadoop/miniconda/bin:$PATH

echo "creating conda environment and configure path to load python from conda"
conda create -c conda-forge -p ~/conda/vision -y python=3.6
export PATH=/home/hadoop/conda/vision/bin:$PATH

echo "This is the path"
echo $PATH

echo "This is the current python interpreter path"
echo $(which python)

echo "success"

mkdir -p vision
cd vision

aws s3 cp s3://stg-relevance-vision-eu-west-1/pipeline/pex/vision_train.ini vision_train.ini
aws s3 cp s3://stg-relevance-vision-eu-west-1/pipeline/pex/vision_train.pex vision_train.pex

export SPARK_HOME=/usr/lib/spark
python3 vision_train.pex -m vision.train.vision_train --config-file vision_train.ini


