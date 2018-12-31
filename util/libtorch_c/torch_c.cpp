#include "torch_c.h"
#include <torch/torch.h>
#include <iostream>

#include <ATen/Functions.h>

//auto tensor = torch::from_blob(
//      v.data(), v.size(), torch::dtype(torch::kFloat64).requires_grad(true));
//
//  bool called = false;
//  {
//    std::vector<int32_t> v = {1, 2, 3};
//    auto tensor = torch::from_blob(
//        v.data(),
//        v.size(),
//        /*deleter=*/[&called](void* data) { called = true; },
//        torch::kInt32);
//  }

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

//auto tensor = torch::from_blob(
//      v.data(), v.size(), torch::dtype(torch::kFloat64).requires_grad(true));

torch::Tensor fromDLPack(const DLTensor* src) {
  at::DeviceType device_type = getATenDeviceType(src->ctx);
  at::ScalarType stype = toScalarType(src->dtype);
  //auto deleter = [src](void * self) {
  //  src->deleter(const_cast<DLManagedTensor*>(src));
  //};
  return torch::from_blob(src->data,
      at::IntList(src->shape, src->ndim),
      at::IntList(src->strides, src->ndim),
      torch::device(device_type).dtype(stype));
}

}

struct ScriptContext {
  std::shared_ptr<torch::jit::script::Module> module;
};

extern "C" void torchBasicTest()
{
  torch::Tensor mat = torch::rand({3,3});
  std::cout << mat << std::endl;
}

extern "C" void* torchCompileScript(const char* script)
{
  ScriptContext* ctx = new ScriptContext();
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
                               long nInputs, DLTensor** inputs,
                               long nOutputs, DLTensor** outputs)
{
  ScriptContext* ctx = (ScriptContext*)scriptCtx;

  try {
    torch::jit::script::Method& method = ctx->module->get_method(fnName);

    torch::jit::Stack stack;

    for (int i=0; i<nInputs; i++) {
      DLTensor* input = inputs[i];
      torch::Tensor tensor = fromDLPack(input);
      stack.push_back(tensor);
    }

    method.run(stack);

    if (stack.size() != nOutputs) {
      // throw std::runtime_error(std::string("Function ") + fnName + " returned " + stack.size() + " elements, expected " + nOutputs);
      throw std::runtime_error(std::string("Function returned unexpected number of outputs - ") + fnName);
    }

    for (int i=0; i<nOutputs; i++) {
      // TODO: convert backwards from IValue to DLTensor (avoid converting non-tensor)
      // Here we should copy values
      // stack[i];
      // outputs[i] = ;
      std::cout<<stack[i]<<std::endl;
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


