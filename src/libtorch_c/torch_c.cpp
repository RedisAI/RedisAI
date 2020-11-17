#include "torch_c.h"
#include "torch/torch.h"
#include "torch/csrc/jit/serialization/import.h"
#include "torch/csrc/jit/api/compilation_unit.h"
#include "ATen/Functions.h"

#include <iostream>
#include <sstream>


namespace {

static DLDataType getDLDataType(const at::Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case at::ScalarType::Byte:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case at::ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Bool:
      throw std::logic_error("Bool is not supported by dlpack");
    case at::ScalarType::BFloat16:
      throw std::logic_error("BFloat16 is not supported by dlpack");
    case at::ScalarType::QInt8:
      throw std::logic_error("QInt8 is not supported by dlpack");
    case at::ScalarType::QUInt8:
      throw std::logic_error("QUInt8 is not supported by dlpack");
    case at::ScalarType::QInt32:
      throw std::logic_error("QInt32 is not supported by dlpack");
    case at::ScalarType::ComplexHalf:
      throw std::logic_error("ComplexHalf is not supported by dlpack");
    case at::ScalarType::ComplexFloat:
      throw std::logic_error("ComplexFloat is not supported by dlpack");
    case at::ScalarType::ComplexDouble:
      throw std::logic_error("ComplexDouble is not supported by dlpack");
    case at::ScalarType::Undefined:
      throw std::logic_error("Undefined is not a valid ScalarType");
    case at::ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}

static DLContext getDLContext(const at::Tensor& tensor, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (tensor.is_cuda()) {
    ctx.device_type = DLDeviceType::kDLGPU;
  } else {
    ctx.device_type = DLDeviceType::kDLCPU;
  }
  return ctx;
}

static at::DeviceType getATenDeviceType(DLDeviceType device_type) {
  switch (device_type) {
    case DLDeviceType::kDLCPU:
      return at::DeviceType::CPU;
    case DLDeviceType::kDLGPU:
      return at::DeviceType::CUDA;
    case DLDeviceType::kDLOpenCL:
      return at::DeviceType::OPENCL;
    case DLDeviceType::kDLROCM:
      return at::DeviceType::HIP;
    default:
      throw std::logic_error("Unsupported device_type: " + std::to_string(device_type));
  }
  return at::DeviceType::CPU; // impossible
}

at::ScalarType toScalarType(const DLDataType& dtype) {
  at::ScalarType stype;
  if (dtype.lanes != 1) throw std::logic_error("ATen does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          stype = at::ScalarType::Byte;
          break;
        default:
          throw std::logic_error("Unsupported kUInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          stype = at::ScalarType::Char;
          break;
        case 16:
          stype = at::ScalarType::Short;
          break;
        case 32:
          stype = at::ScalarType::Int;
          break;
        case 64:
          stype = at::ScalarType::Long;
          break;
        default:
          throw std::logic_error("Unsupported kInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          stype = at::ScalarType::Half;
          break;
        case 32:
          stype = at::ScalarType::Float;
          break;
        case 64:
          stype = at::ScalarType::Double;
          break;
        default:
          throw std::logic_error("Unsupported kFloat bits " + std::to_string(dtype.bits));
      }
      break;
    default:
      throw std::logic_error("Unsupported code " + std::to_string(dtype.code));
  }
  return stype;
}

torch::Tensor fromDLPack(const DLTensor* src) {
  at::DeviceType device_type = getATenDeviceType(src->ctx.device_type);
  at::ScalarType stype = toScalarType(src->dtype);
  // torch::Device device(device_type, src->ctx.device_id);
  torch::Device device(device_type, -1);
  // torch::DeviceType device = device_type;
  return torch::from_blob(src->data,
      at::IntArrayRef(src->shape, src->ndim),
      at::IntArrayRef(src->strides, src->ndim),
      torch::device(device).dtype(stype));
}

struct ATenDLMTensor {
  torch::Tensor handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor * arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);
}

DLManagedTensor* toManagedDLPack(const torch::Tensor& src_) {
  ATenDLMTensor * atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src_;
  auto& src = atDLMTensor->handle;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  int64_t device_id = 0;
  if (src.is_cuda()) {
    device_id = src.get_device();
  }
  atDLMTensor->tensor.dl_tensor.ctx = getDLContext(src, device_id);
  atDLMTensor->tensor.dl_tensor.ndim = src.dim();
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(atDLMTensor->tensor);
}

struct ModuleContext {
  std::shared_ptr<torch::jit::script::Module> module;
  std::shared_ptr<torch::jit::script::CompilationUnit> cu;
  DLDeviceType device;
  int64_t device_id;
};

void torchRunModule(ModuleContext* ctx, const char* fnName, int variadic,
                    long nInputs, DLManagedTensor** inputs,
                    long nOutputs, DLManagedTensor** outputs) {
  // Checks device, if GPU then move input to GPU before running
  // TODO: This will need to change at some point, as individual tensors will have their placement
  // and script will only make sure that placement is correct

  torch::DeviceType device_type;
  switch (ctx->device) {
    case kDLCPU:
      device_type = torch::kCPU;
      break;
    case kDLGPU:
      device_type = torch::kCUDA;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported device ") + std::to_string(ctx->device));
  }

  torch::Device device(device_type, ctx->device_id);

  torch::jit::Stack stack;

  for (int i=0; i<nInputs; i++) {
    if (i == variadic) {
      break;
    }
    DLTensor* input = &(inputs[i]->dl_tensor);
    torch::Tensor tensor = fromDLPack(input);
    stack.push_back(tensor.to(device));
  }

  if (variadic != -1 ) {
    std::vector<torch::Tensor> args;
    for (int i=variadic; i<nInputs; i++) {
      DLTensor* input = &(inputs[i]->dl_tensor);
      torch::Tensor tensor = fromDLPack(input);
      tensor.to(device);
      args.emplace_back(tensor);
    }
    stack.push_back(args);
  }

  if (ctx->module) {
    torch::NoGradGuard guard;
    torch::jit::script::Method method = ctx->module->get_method(fnName);
    method.run(stack);
  }
  else {
    torch::NoGradGuard guard;
    torch::jit::Function& fn = ctx->cu->get_function(fnName);
    fn.run(stack);
  }

  torch::DeviceType output_device_type = torch::kCPU;
  torch::Device output_device(output_device_type, -1);

  int count = 0;
  for (size_t i=0; i<stack.size(); i++) {
    if (count > nOutputs-1) {
      throw std::runtime_error(std::string("Function returned unexpected number of outputs - ") + fnName);
    }

    if (stack[i].isTensor()) {
      outputs[count++] = toManagedDLPack(stack[i].toTensor().contiguous().to(output_device));
    }
    else if (stack[i].isTensorList()) {
      auto list = stack[i].toTensorList();
      for (size_t j=0; j<list.size(); j++) {
        outputs[count++] = toManagedDLPack(list.get(j).contiguous().to(output_device));
      }
    }
    else if (stack[i].isTuple()) {
      auto& elements = stack[i].toTuple()->elements();
      for (size_t j=0; j<elements.size(); j++) {
        if (elements[j].isTensor()) {
          outputs[count++] = toManagedDLPack(elements[j].toTensor().contiguous().to(output_device));
        }
        else {
          throw std::runtime_error(std::string("Function returned non-tensor values") + fnName);
        }
      }
    }
    else {
      throw std::runtime_error(std::string("Function returned non-tensor values") + fnName);
    }
  }

  if (count != nOutputs) {
    throw std::runtime_error(std::string("Function returned unexpected number of outputs - ") + fnName);
  }
}

}

extern "C" void torchBasicTest()
{
  torch::Tensor mat = torch::rand({3,3});
  std::cout << mat << std::endl;
}

extern "C" DLManagedTensor* torchNewTensor(DLDataType dtype, long ndims, int64_t* shape, int64_t* strides, char* data)
{
  // at::DeviceType device_type = getATenDeviceType(kDLCPU);
  at::ScalarType stype = toScalarType(dtype);
  torch::Device device(getATenDeviceType(kDLCPU), -1);
  torch::Tensor tensor = torch::from_blob(data,
      at::IntArrayRef(shape, ndims),
      at::IntArrayRef(strides, ndims),
      // torch::device(at::DeviceType::CPU).dtype(stype));
      torch::device(device).dtype(stype));

  DLManagedTensor *dl_tensor = toManagedDLPack(tensor);

  return dl_tensor;
}

extern "C" void* torchCompileScript(const char* script, DLDeviceType device, int64_t device_id,
                                    char **error, void* (*alloc)(size_t))
{
  ModuleContext* ctx = new ModuleContext();
  ctx->device = device;
  ctx->device_id = device_id;
  try {
    auto cu = torch::jit::compile(script);
    auto aten_device_type = getATenDeviceType(device);
    if (aten_device_type == at::DeviceType::CUDA && !torch::cuda::is_available()) {
      throw std::logic_error("GPU requested but Torch couldn't find CUDA");
    }
    ctx->cu = cu;
    ctx->module = nullptr;
  }
  catch(std::exception& e) {
    size_t len = strlen(e.what()) +1;
    *error = (char*)alloc(len * sizeof(char));
    strcpy(*error, e.what());
    (*error)[len-1] = '\0';
    delete ctx;
    return NULL;
  }
  return ctx;
}

extern "C" void* torchLoadModel(const char* graph, size_t graphlen, DLDeviceType device, int64_t device_id,
                                char **error, void* (*alloc)(size_t))
{
  std::string graphstr(graph, graphlen);
  std::istringstream graph_stream(graphstr, std::ios_base::binary);
  ModuleContext* ctx = new ModuleContext();
  ctx->device = device;
  ctx->device_id = device_id;
  try {
    // TODO: move to device now
    auto module = std::make_shared<torch::jit::script::Module>(torch::jit::load(graph_stream));
    auto aten_device_type = getATenDeviceType(device);
    if (aten_device_type == at::DeviceType::CUDA && !torch::cuda::is_available()) {
      throw std::logic_error("GPU requested but Torch couldn't find CUDA");
    }
    torch::Device aten_device(aten_device_type, device_id);
    module->to(aten_device);
    ctx->module = module;
    ctx->cu = nullptr;
  }
  catch(std::exception& e) {
    size_t len = strlen(e.what()) +1;
    *error = (char*)alloc(len * sizeof(char));
    strcpy(*error, e.what());
    (*error)[len-1] = '\0';
    // delete ctx;
    return NULL;
  }
  return ctx;
}

extern "C" void torchRunScript(void* scriptCtx, const char* fnName, int variadic,
                               long nInputs, DLManagedTensor** inputs,
                               long nOutputs, DLManagedTensor** outputs,
                               char **error, void* (*alloc)(size_t))
{
  ModuleContext* ctx = (ModuleContext*)scriptCtx;
  try {
    torchRunModule(ctx, fnName, variadic, nInputs, inputs, nOutputs, outputs);
  }
  catch(std::exception& e) {
    size_t len = strlen(e.what()) +1;
    *error = (char*)alloc(len * sizeof(char));
    strcpy(*error, e.what());
    (*error)[len-1] = '\0';
  }
}

extern "C" void torchRunModel(void* modelCtx,
                              long nInputs, DLManagedTensor** inputs,
                              long nOutputs, DLManagedTensor** outputs,
                              char **error, void* (*alloc)(size_t))
{
  ModuleContext* ctx = (ModuleContext*)modelCtx;
  try {
    torchRunModule(ctx, "forward", -1, nInputs, inputs, nOutputs, outputs);
  }
  catch(std::exception& e) {
    size_t len = strlen(e.what()) +1;
    *error = (char*)alloc(len * sizeof(char));
    strcpy(*error, e.what());
    (*error)[len-1] = '\0';
  }
}

extern "C" void torchSerializeModel(void* modelCtx, char **buffer, size_t *len,
                                    char **error, void* (*alloc)(size_t))
{
  ModuleContext* ctx = (ModuleContext*)modelCtx;
  std::ostringstream out;
  try {
    ctx->module->save(out);
    auto out_str = out.str();
    int size = out_str.size();
    *buffer = (char *)alloc(size);
    memcpy(*buffer, out_str.c_str(), size);
    *len = size;
  }
  catch(std::exception& e) {
    size_t len = strlen(e.what()) +1;
    *error = (char*)alloc(len * sizeof(char));
    strcpy(*error, e.what());
    (*error)[len-1] = '\0';
  }
}

extern "C" void torchDeallocContext(void* ctx)
{
  ModuleContext* ctx_ = (ModuleContext*)ctx;
  if (ctx_) {
    delete ctx_;
  }
}
