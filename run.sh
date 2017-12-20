#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
ROOT_DIR=$SCRIPT_PATH
DATA_DIR=$ROOT_DIR/data
BUILD_DIR=$ROOT_DIR/build

echo "BUILD..."
mkdir -p $BUILD_DIR
cd $BUILD_DIR && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4
cd $ROOT_DIR
echo "RUN..."
$BUILD_DIR/bin/inception3_basic -u $DATA_DIR/inception_v3.uff -i $DATA_DIR/panda.ppm -l $DATA_DIR/imagenet_slim_labels.txt
