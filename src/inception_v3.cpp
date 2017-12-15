#include "common.h"
#include "inception_v3.h"
#include <chrono>
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>

static Logger gLogger;
#define MAX_WORKSPACE 0

InceptionV3::InceptionV3(
  const std::string& uff_fn,
  const std::string& src_node_name,
  const std::string& dst_node_name,
  int src_n, /* input batch   */
  int src_c, /* input channel */
  int src_h, /* input height  */
  int src_w, /* input width   */
  bool enable_fp16,
  bool enable_nhwc)
{
  // initialization
  src_dims = nvinfer1::DimsCHW(src_c, src_h, src_w);
  max_batch = src_n;
  parser = nvuffparser::createUffParser();
  assert(parser);
  parser->registerInput(src_node_name.c_str(), src_dims,
    (enable_nhwc) ? nvuffparser::UffInputOrder::kNHWC :
    nvuffparser::UffInputOrder::kNCHW);
  parser->registerOutput(dst_node_name.c_str());
  engine = create_engine(uff_fn.c_str(), src_n, *parser, enable_fp16);
  assert(engine);
  context = engine->createExecutionContext();
  assert(context);
  // after creating engine, parser is no-use (?
  parser->destroy();
  parser = nullptr;
  // buffer allocation
  buffs = create_buffers(*engine, src_n);
}

InceptionV3::~InceptionV3(void)
{
  release_buffers(buffs);
  if (context)  context->destroy();
  if (parser)   parser->destroy();
  if (engine)   engine->destroy();
  nvuffparser::shutdownProtobufLibrary();
}

uint64_t InceptionV3::get_src_count(void)
{
  return buffs[src_buff_idx].count / max_batch;
}

uint64_t InceptionV3::get_dst_count(void)
{
  return buffs[dst_buff_idx].count / max_batch;
}

void InceptionV3::set_src_buffer(const std::vector<float>& host_buff, int batch)
{
  assert(batch <= max_batch);
  auto src_count = host_buff.size();
  auto copy_count = this->get_src_count() * batch;
  assert(src_count >= copy_count);
  const void* host_ptr = reinterpret_cast<const void*>(host_buff.data());
  auto copy_size = copy_count * sizeof(float);
  auto& ref_buff = buffs[src_buff_idx];
  assert(copy_size <= ref_buff.size);
  auto error = cudaMemcpy(ref_buff.ptr, host_ptr, copy_size, cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);
}

float InceptionV3::inference(int batch)
{
  assert(batch <= max_batch);
  std::vector<void*> ptrs;
  for (auto& buff : buffs)
  {
    ptrs.push_back(buff.ptr);
  }
  auto start = std::chrono::high_resolution_clock::now();
  auto ok = context->execute(batch, reinterpret_cast<void**>(ptrs.data()));
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration<float, std::milli>(end - start).count();
  assert(ok == true);
  return ms;
}

void InceptionV3::get_dst_buffer(std::vector<float>& host_buff, int batch)
{
  assert(batch <= max_batch);
  auto& ref_buff = buffs[dst_buff_idx];
  auto copy_count = this->get_dst_count() * batch;
  assert(host_buff.size() >= copy_count);
  auto copy_size  = copy_count * sizeof(float);
  assert(copy_size <= ref_buff.size);
  auto error = cudaMemcpy(host_buff.data(), ref_buff.ptr, copy_size, cudaMemcpyDeviceToHost);
  assert(error == cudaSuccess);
}

nvinfer1::ICudaEngine* InceptionV3::create_engine(const char* uff_fn,
  int max_batch, nvuffparser::IUffParser& parser, bool enable_fp16)
{
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  auto data_type = nvinfer1::DataType::kFLOAT;
  if (enable_fp16) {
    data_type = nvinfer1::DataType::kHALF;
  }
  auto ok = parser.parse(uff_fn, *network, data_type);
  assert(ok == true);
  builder->setHalf2Mode(enable_fp16);
  builder->setMaxBatchSize(max_batch);
  builder->setMaxWorkspaceSize(MAX_WORKSPACE);
  nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);
  network->destroy();
  builder->destroy();
  return engine;
}

std::vector<InceptionV3::InferBuff> InceptionV3::create_buffers(
  const nvinfer1::ICudaEngine& engine, int batch)
{
  std::vector<InceptionV3::InferBuff> buffs;
  auto nb_bindings = engine.getNbBindings();
  assert(nb_bindings == 2); // inception v3 should only need 1 input and 1 output
  for (auto i = 0; i < nb_bindings; ++i)
  {
    if ( engine.bindingIsInput(i) ) {
      src_buff_idx = i;
    }
    else {
      dst_buff_idx = i;
    }
    InceptionV3::InferBuff buff;
    auto dims  = engine.getBindingDimensions(i);
    buff.type  = engine.getBindingDataType(i);
    buff.count = volume(dims) * batch;
    buff.size = buff.count * type_size(buff.type);
    auto error = cudaMalloc(&buff.ptr, buff.size);
    assert(error == cudaSuccess);
    error = cudaMemset(buff.ptr, 0, buff.size);
    assert(error == cudaSuccess);
    buffs.push_back(buff);
  }
  return buffs;
}

void InceptionV3::release_buffers(std::vector<InceptionV3::InferBuff>& buffs)
{
  for (auto& buff : buffs)
  {
    if (buff.ptr) {
      cudaFree(buff.ptr);
    }
  }
}

inline uint64_t InceptionV3::volume(const nvinfer1::Dims& d)
{
  uint64_t v = 1;
  for (int64_t i = 0; i < d.nbDims; i++)
    v *= d.d[i];
  return v;
}

inline unsigned int InceptionV3::type_size(nvinfer1::DataType t)
{
  switch (t)
  {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
  }
  assert(0);
  return 0;
}
