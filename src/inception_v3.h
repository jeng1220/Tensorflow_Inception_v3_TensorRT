#ifndef INCEPTION_V3_H_
#define INCEPTION_V3_H_

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include <string>
#include <vector>

class InceptionV3
{
public:
  InceptionV3(const std::string& uff_fn,
    const std::string& src_node_name,
    const std::string& dst_node_name,
    int src_n, /* input batch   */
    int src_c, /* input channel */
    int src_h, /* input height  */
    int src_w, /* input width   */
    bool enable_fp16 = false,
    bool enable_nhwc = false);

  ~InceptionV3(void);

  uint64_t get_src_count(void);

  uint64_t get_dst_count(void);

  void set_src_buffer(const std::vector<float>& host_buff, int batch);

  float inference(int batch);

  void get_dst_buffer(std::vector<float>& host_buff, int batch);

private:
  struct InferBuff {
    void* ptr      = nullptr;
    uint64_t count = 0;
    size_t size    = 0;
    nvinfer1::DataType type;
  };

  std::vector<InferBuff> m_buffs;
  nvuffparser::IUffParser* m_parser      = nullptr;
  nvinfer1::ICudaEngine* m_engine        = nullptr;
  nvinfer1::IExecutionContext* m_context = nullptr;
  nvinfer1::DimsCHW m_src_dims;
  int m_src_buff_idx = 0;
  int m_dst_buff_idx = 0;
  int m_max_batch    = 0;

  nvinfer1::ICudaEngine* create_engine(const char* uff_fn,
    int max_batch, nvuffparser::IUffParser& parser, bool enable_fp16);

  std::vector<InferBuff> create_buffers(const nvinfer1::ICudaEngine& engine, int batch);

  void release_buffers(std::vector<InferBuff>& buffs);

  inline uint64_t volume(const nvinfer1::Dims& d);

  inline unsigned int type_size(nvinfer1::DataType t);

};

#endif
