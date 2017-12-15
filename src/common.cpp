#include "common.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg)
{
  // suppress info-level messages
  if (severity == Severity::kINFO) return;
  switch (severity)
  {
    case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
    case Severity::kERROR:          std::cerr << "ERROR: ";          break;
    case Severity::kWARNING:        std::cerr << "WARNING: ";        break;
    case Severity::kINFO:           std::cerr << "INFO: ";           break;
    default:                        std::cerr << "UNKNOWN: ";        break;
  }
  std::cerr << msg << std::endl;
}

void InplaceConvertRGBtoBGR(std::vector<uint8_t>& src)
{
  auto count = src.size();
  assert(count % 3 == 0);
  for (size_t i = 0; i < count; i += 3)
  {
    auto tmp = src[i];
    size_t idx = i + 2;
    // swap
    src[i] = src[idx];
    src[idx] = tmp;
  }
}

std::vector<uint8_t> ConvertHWCtoCHW(const std::vector<uint8_t>& src,
  int batch, int h, int w, int channel)
{
  auto count = src.size();
  assert(count == static_cast<size_t>(batch) * h * w * channel);
  std::vector<uint8_t> dst(count);

  const uint8_t* src_ptr = src.data();

  for (int n = 0; n < batch; ++n)
  {
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        for (int c = 0; c < channel; ++c)
        {
          size_t idx = n * (h*w*channel)
            + c * (h*w)
            + y * (w)
            + x;
          dst[idx] = *src_ptr;
          src_ptr++;
        }
      }
    }
  }
  return dst;
}

std::vector<uint8_t> ReadPPMFile(const std::string& fn,
  bool convert_bgr, bool convert_plane)
{
  std::ifstream fs;
  fs.open(fn, std::ifstream::in | std::ifstream::binary);
  assert(fs.is_open());
  std::string magic, h, w, max;
  fs >> magic;
  assert(magic.compare("P6") == 0);
  fs >> h;
  assert(h.compare("299")    == 0);
  fs >> w;
  assert(w.compare("299")    == 0);
  fs >> max;
  assert(max.compare("255")  == 0);
  auto i32h = std::stoi(h);
  auto i32w = std::stoi(w);
  size_t i8count = 3 * i32h * i32w;
  std::vector<uint8_t> i8buff(i8count);
  fs.seekg(1, fs.cur);
  fs.read(reinterpret_cast<char*>(i8buff.data()), i8count);

  if (convert_bgr) {
    InplaceConvertRGBtoBGR(i8buff);
  }

  if (convert_plane) {
    return ConvertHWCtoCHW(i8buff,
      1, i32h, i32w, 3);
  }

  return i8buff;
}

std::vector<float> ImagePreprocess(const std::vector<uint8_t>& src,
  float img_mean, float img_std, bool auto_mean)
{
  auto count = src.size();
  std::vector<float> dst(count);
  float sub_mean = img_mean;
  if (auto_mean) {
    double accum = 0;
    for (auto v : src)
    {
      accum += static_cast<double>(v);
    }
    sub_mean = static_cast<float>(accum / static_cast<double>(count));
  }
  for (size_t i = 0; i < count; ++i)
  {
    dst[i] = (static_cast<float>(src[i]) - sub_mean) / img_std;
  }
  return dst;
}

std::vector<std::string> ReadLabelFile(const std::string& fn)
{
  std::ifstream ifs;
  ifs.open(fn, std::ifstream::in);
  assert(ifs.is_open());
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(ifs, line)) {
    labels.push_back(line);
  }
  assert(labels.size() == 1001);
  return labels;
}

bool SortFunc(const InferPack& a, const InferPack& b)
{
  return (a.prob > b.prob);
}

std::vector<InferPack> SortInferResult(
  const std::vector<float>& probs,
  const std::vector<std::string>& labels,
  int run_batch)
{
  assert(probs.size() == labels.size());
  auto count = probs.size();
  std::vector<InferPack> packs(count);
  for (size_t i = 0; i < count; ++i)
  {
    packs[i].prob  = probs[i];
    packs[i].label = labels[i];
  }
  std::sort(packs.begin(), packs.end(), SortFunc);
  return packs;
}

void PrintToFive(const std::vector<InferPack>& src)
{
  assert(src.size() > 5);
  std::cout << "Top 5:\n";
  for (int i = 0; i < 5; ++i)
  {
    auto& pack = src[i];
    std::cout << i << ", " << pack.label << ", " << pack.prob << "\n";
  }
  std::cout << std::endl;
}