#include "backends/torch.h"
#include "tensor.h"
#include "utils/arr_rm_alloc.h"
#include "torch_c.h"

#if 0
TF_DataType RAI_GetTFDataTypeFromDL(DLDataType dtype) {

  return 0;
}

DLDataType RAI_GetDLDataTypeFromTF(TF_DataType dtype) {
  switch (dtype) {
    case TF_FLOAT:
      return (DLDataType){ .code = kDLFloat, .bits = 32, .lanes = 1 };
    case TF_DOUBLE:
      return (DLDataType){ .code = kDLFloat, .bits = 64, .lanes = 1 };
    case TF_INT8:
      return (DLDataType){ .code = kDLInt, .bits = 8, .lanes = 1 };
    case TF_INT16:
      return (DLDataType){ .code = kDLInt, .bits = 16, .lanes = 1 };
    case TF_INT32:
      return (DLDataType){ .code = kDLInt, .bits = 32, .lanes = 1 };
    case TF_INT64:
      return (DLDataType){ .code = kDLInt, .bits = 64, .lanes = 1 };
    case TF_UINT8:
      return (DLDataType){ .code = kDLUInt, .bits = 8, .lanes = 1 };
    case TF_UINT16:
      return (DLDataType){ .code = kDLUInt, .bits = 16, .lanes = 1 };
    default:
      return (DLDataType){ .bits = 0 };
  }
  return (DLDataType){ .bits = 0 };
}

//RAI_Tensor* RAI_TensorCreateFromTorchTensor(TF_Tensor *tensor) {
RAI_Tensor* RAI_TensorCreateFromTorchTensor(void *tensor) {
  // ret->tensor = (DLTensor){
  //     .ctx = ctx,
  //     .data = data,
  //     .ndim = ndims,
  //     .dtype = RAI_GetDLDataTypeFromTF(TF_TensorType(tensor)),
  //     .shape = shape,
  //     .strides = NULL,
  //     .byte_offset = 0
  // };

  // ret->refCount = 1;
  // return ret;
  return NULL;
}

void RAI_TFDeallocator(void* data, size_t len, void* arg) {
  // printf("DEALLOCATOR CALLED\n");
  // do nothing, memory is managed by Redis
}

// TF_Tensor* RAI_TorchTensorFromTensor(RAI_Tensor* t){
void* RAI_TorchTensorFromTensor(RAI_Tensor* t){
  // return TF_NewTensor(
  //     RAI_GetTFDataTypeFromDL(t->tensor.dtype),
  //     t->tensor.shape,
  //     t->tensor.ndim,
  //     t->tensor.data,
  //     RAI_TensorByteSize(t),
  //     &RAI_TFDeallocator,
  //     NULL);
  return NULL;
}

#endif

RAI_Script *RAI_ScriptCreateTorch(RAI_Device device, const char *scriptdef) {

  size_t scriptlen = strlen(scriptdef);
  char* scriptdef_ = RedisModule_Alloc(scriptlen * sizeof(char));
  memcpy(scriptdef_, scriptdef, scriptlen);

  DLDeviceType dl_device;
  switch (device) {
    case RAI_DEVICE_CPU:
      dl_device = kDLCPU;
      break;
    case RAI_DEVICE_GPU:
      dl_device = kDLGPU;
      break;
    default:
      // TODO error unsupported device
      break;
  }

  void* script = torchCompileScript(scriptdef, dl_device);

  RAI_Script* ret = RedisModule_Alloc(sizeof(*ret));
  ret->script = script;
  ret->scriptdef = scriptdef_;
  ret->device = device;
  ret->refCount = 1;

  return ret;
}

void RAI_ScriptFreeTorch(RAI_Script* script) {

  torchDeallocScript(script->script);
  RedisModule_Free(script->scriptdef);
  RedisModule_Free(script);
}

int RAI_ScriptRunTorch(RAI_ScriptRunCtx* sctx) {

  long nInputs = array_len(sctx->inputs);
  long nOutputs = array_len(sctx->outputs);

  DLManagedTensor* inputs[nInputs];
  DLManagedTensor* outputs[nOutputs];

  for (size_t i=0; i<nInputs; i++) {
    inputs[i] = &sctx->inputs[i].tensor->tensor;
  }

  for (size_t i=0; i<nOutputs; i++) {
    outputs[i] = &sctx->outputs[i].tensor->tensor;
  }

  long ret = torchRunScript(sctx->script->script, sctx->fnname, nInputs, inputs, nOutputs, outputs);

  for (size_t i=0; i<nOutputs; i++) {
    sctx->outputs[0].tensor = RAI_TensorCreateFromDLTensor(outputs[i]);
  }

  return ret;
}
