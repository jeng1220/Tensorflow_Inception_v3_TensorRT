#ifndef COMMON_H_
#define COMMON_H_
#include "NvInfer.h"
#include <cstdint>
#include <string>
#include <vector>

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
  public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override;
};

std::vector<uint8_t> ReadPPMFile(const std::string& fn,
  bool convert_bgr = true, bool convert_plane = true);

std::vector<float> ImagePreprocess(const std::vector<uint8_t>& src,
  float img_mean, float img_std, bool auto_mean = false);

std::vector<std::string> ReadLabelFile(const std::string& fn);

struct InferPack {
  float prob = 0;
  std::string label;
};

std::vector<InferPack> SortInferResult(
  const std::vector<float>& probs,
  const std::vector<std::string>& labels,
  int run_batch = 1);

void PrintToFive(const std::vector<InferPack>& src);

#endif // COMMON_H_
