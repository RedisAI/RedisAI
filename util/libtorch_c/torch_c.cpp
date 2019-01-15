#include "torch_c.h"
#include <torch/torch.h>
#include <iostream>

#include <ATen/Functions.h>

namespace {

static DLDataType getDLDataType(const at::Type& type) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = type.elementSizeInBytes() * 8;
  switch (type.scalarType()) {
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

static DLContext getDLContext(const at::Type& type, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (type.is_cuda()) {
    ctx.device_type = DLDeviceType::kDLGPU;
  } else {
    ctx.device_type = DLDeviceType::kDLCPU;
  }
  return ctx;
}

static at::DeviceType getATenDeviceType(const DLContext& ctx) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      return at::DeviceType::CPU;
    case DLDeviceType::kDLGPU:
      return at::DeviceType::CUDA;
    case DLDeviceType::kDLOpenCL:
      return at::DeviceType::OPENCL;
    case DLDeviceType::kDLROCM:
      return at::DeviceType::HIP;
    default:
      throw std::logic_error("Unsupported device_type: " + std::to_string(ctx.device_type));
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
  at::DeviceType device_type = getATenDeviceType(src->ctx);
  at::ScalarType stype = toScalarType(src->dtype);
  return torch::from_blob(src->data,
      at::IntList(src->shape, src->ndim),
      at::IntList(src->strides, src->ndim),
      torch::device(device_type).dtype(stype));
}

struct ATenDLMTensor {
  torch::Tensor handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor * arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);
}

// TODO: we need to retain the input tensor in a backend_context
// a struct with a shared_ptr for the tensor? and
// make it go out of scope upon deallocation
DLManagedTensor* toManagedDLPack(const torch::Tensor& src) {
  ATenDLMTensor * atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  int64_t device_id = 0;
  if (src.is_cuda()) {
    device_id = src.get_device();
  }
  atDLMTensor->tensor.dl_tensor.ctx = getDLContext(src.type(), device_id);
  atDLMTensor->tensor.dl_tensor.ndim = src.dim();
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src.type());
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(atDLMTensor->tensor);
}
//  DLManagedTensor* dl_tensor = new DLManagedTensor;
//  dl_tensor->dl_tensor.data = src.data_ptr();
//  int64_t device_id = 0;
//  if (src.is_cuda()) {
//    device_id = src.get_device();
//  }
//  dl_tensor->dl_tensor.ctx = getDLContext(src.type(), device_id);
//  dl_tensor->dl_tensor.ndim = src.dim();
//  dl_tensor->dl_tensor.dtype = getDLDataType(src.type());
//  dl_tensor->dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());
//  dl_tensor->dl_tensor.strides = const_cast<int64_t*>(src.strides().data());
//  dl_tensor->dl_tensor.byte_offset = 0;
//  torch::Tensor* manager_ctx = new torch::Tensor(src);
//  dl_tensor->manager_ctx = manager_ctx;
//  dl_tensor->deleter = deleter;
//  return dl_tensor;


}

struct ScriptContext {
  std::shared_ptr<torch::jit::script::Module> module;
  DLDeviceType device;
};

extern "C" void torchBasicTest()
{
  torch::Tensor mat = torch::rand({3,3});
  std::cout << mat << std::endl;
}

extern "C" void* torchCompileScript(const char* script, DLDeviceType device)
{
  ScriptContext* ctx = new ScriptContext();
  ctx->device = device;
  try {
    auto module = torch::jit::compile(script);
    ctx->module = module;
  }
  catch(std::exception& e) {
    std::cout << e.what() << std::endl;
    return NULL;
  }
  return ctx;
}

extern "C" long torchRunScript(void* scriptCtx, const char* fnName,
                               long nInputs, DLManagedTensor** inputs,
                               long nOutputs, DLManagedTensor** outputs)
{
  ScriptContext* ctx = (ScriptContext*)scriptCtx;

  // TODO: check device, if GPU move input before running
  // This will need to change at some point, as individual tensors will have their placement
  // and script will only make sure that placement is correct

  torch::DeviceType device;
  switch (ctx->device) {
    case kDLCPU:
      device = torch::kCPU;
      break;
    case kDLGPU:
      device = torch::kCUDA;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported device ") + std::to_string(ctx->device));
  }

  try {
    torch::jit::script::Method& method = ctx->module->get_method(fnName);

    torch::jit::Stack stack;

    for (int i=0; i<nInputs; i++) {
      DLTensor* input = &(inputs[i]->dl_tensor);
      torch::Tensor tensor = fromDLPack(input);
      stack.push_back(tensor.to(device));
    }

    method.run(stack);

    if (stack.size() != nOutputs) {
      throw std::runtime_error(std::string("Function returned unexpected number of outputs - ") + fnName);
    }

    for (int i=0; i<nOutputs; i++) {
      // TODO: convert backwards from IValue to DLTensor (avoid converting non-tensor)
      // stack[i];
      // TODO: what about isTensorList?
      // TODO: move to target device
      if (stack[i].isTensor()) {
        torch::Tensor tensor = stack[i].toTensor();
        outputs[i] = toManagedDLPack(tensor);
      }
    }
  }
  catch(std::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}

extern "C" void torchDeallocScript(void* scriptCtx)
{
  ScriptContext* ctx = (ScriptContext*)scriptCtx;
  if (ctx) {
    delete ctx;
  }
}


