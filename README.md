This is a simple demonstration for running Tensorflow inception v3 model on TensorRT

### Requirement ###
* TensorRT 3.0 GA
* cuDNN 7 for CUDA 9.0
* Cmake 3.8 (natively support CUDA, refer to "https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake/")


### Build ###
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j4
```

### Run ###
```shell
$ ./build/bin/inception3_basic -h
$ ./build/bin/inception3_basic -u ./data/inception_v3.uff -i ./data/panda.ppm -l ./data/imagenet_slim_labels.txt
```

### Appendix ###

#### Prepare Model and Label ####
The network model, label and images (PPM format) have been prepared at data folder.
Following text shows where and how to get these files.
1. Refer to "https://www.tensorflow.org/tutorials/image_recognition" and download Tensorflow inception v3 model.
```shell
$ wget "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
$ tar zxvf inception_v3_2016_08_28_frozen.pb.tar.gz
```
2. Then, follow instructions from TensorRT User Guide to convert pb file to UFF file.

#### Prepare PPM Image ####
PPM images as input image format can be referred to "https://en.wikipedia.org/wiki/Netpbm_format" .
The default input image size of inception v3 is width=299, height=299 and channel=3.
1. kolourpaint, a free software can resize image and convert JPEG, PNG image to PPM format.
