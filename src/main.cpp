#include "common.h"
#include "inception_v3.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

int main(int argc, char** argv)
{
  std::string uff_fn       = "inception_v3.uff";
  std::string input_layer  = "input";
  std::string output_layer = "InceptionV3/Predictions/Reshape_1";
  std::string image_fn     = "grace_hopper.ppm";
  std::string label_fn     = "imagenet_slim_labels.txt";
  int input_channel = 3;
  int input_width   = 299;
  int input_height  = 299;
  float input_mean  = 0.0f;
  float input_std   = 255.0f;
  bool convert_to_bgr   = false;
  bool enable_nhwc      = false;
  bool enable_fp16      = false;

  int c;
  while ((c = getopt(argc, argv, "u:i:l:m:d:tpfh")) != -1)
  {
    switch (c)
    {
      case 'u':
        uff_fn = std::string(optarg);   break;
      case 'i':
        image_fn = std::string(optarg); break;
      case 'l':
        label_fn = std::string(optarg); break;
      case 'm':
        input_mean = std::atof(optarg); break;
      case 'd':
        input_std = std::atof(optarg);  break;
      case 't':
        convert_to_bgr = true;          break;
      case 'p':
        enable_nhwc = true;             break;
      case 'f':
        enable_fp16 = true;             break;
      case 'h':
        std::cout <<
          "-u [string], input NVIDIA UFF\n"
          "-i [string], input image with PPM format\n"
          "-l [string], input label\n"
          "-m [float], input mean, default 0.0f\n"
          "-d [float], input std, default 255.0f\n"
          "-t, convert image from RGB to BGR format, default is false\n"
          "-p, use image format as interlance mode, default is false\n"
          "-f, enable FP16 mode, default is false\n"
          "-h, show this info\n"
          << std::endl;
        return EXIT_SUCCESS;
      case '?':
        std::cout << "invalid parameter" << std::endl;
        return EXIT_FAILURE;
    }
  }

  // initial TensorRT
  int max_batch = 1;
  InceptionV3 net(uff_fn,
    input_layer, output_layer,
    max_batch, input_channel, input_width, input_height,
    enable_fp16, enable_nhwc);

  // read image
  auto img_buff = ReadPPMFile(image_fn, convert_to_bgr,
    !enable_nhwc);
  auto src_buff = ImagePreprocess(img_buff,
    input_mean, input_std);

  // copy input from host buffer to device buffer
  int run_batch = 1;
  net.set_src_buffer(src_buff, run_batch);

  // inference (forward propagation)
  auto time_ms = net.inference(run_batch);
  #if 1
  std::cout << "batch, " << run_batch << ", "
            << "time, "  << time_ms   << ", ms"
            << std::endl;
  #endif

  // host buffer allocation
  auto dst_count = run_batch * net.get_dst_count();
  std::vector<float> dst_buff(dst_count);

  // copy result from device buffer to host buffer
  net.get_dst_buffer(dst_buff, run_batch);

  // show top five
  std::vector<std::string> labels =
    ReadLabelFile(label_fn);
  auto results = SortInferResult(dst_buff, labels, run_batch);
  PrintToFive(results);
  return EXIT_SUCCESS;
}
