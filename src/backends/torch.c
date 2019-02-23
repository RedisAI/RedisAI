#include "backends/torch.h"
#include "tensor.h"
#include "utils/alloc.h"
#include "utils/arr_rm_alloc.h"
#include "torch_c.h"


RAI_Model *RAI_ModelCreateTorch(RAI_Backend backend, RAI_Device device,
                                const char *modeldef, size_t modellen,
                                RAI_Error *err) {
  DLDeviceType dl_device;
  switch (device) {
    case RAI_DEVICE_CPU:
      dl_device = kDLCPU;
      break;
    case RAI_DEVICE_GPU:
      dl_device = kDLGPU;
      break;
    default:
      RAI_SetError(err, RAI_EMODELCONFIGURE, "Error configuring model: unsupported device\n");
      return NULL;
  }

  char* err_descr = NULL;
  void* model = torchLoadModel(modeldef, modellen, dl_device, &err_descr);

  if (model == NULL) {
    RAI_SetError(err, RAI_EMODELCREATE, err_descr);
    free(err_descr);
    return NULL;
  }

  RAI_Model* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->model = model;
  ret->session = NULL;
  ret->backend = backend;
  ret->inputs = NULL;
  ret->outputs = NULL;
  ret->refCount = 1;

  return ret;
}

void RAI_ModelFreeTorch(RAI_Model* model, RAI_Error *err) {
  torchDeallocContext(model->model);
}

int RAI_ModelRunTorch(RAI_ModelRunCtx* mctx, RAI_Error *err) {

  DLManagedTensor** inputs = RedisModule_Calloc(1, sizeof(*inputs));
  DLManagedTensor** outputs = RedisModule_Calloc(1, sizeof(*outputs));

  for (size_t i=0 ; i<array_len(mctx->inputs); ++i) {
    inputs[i] = &mctx->inputs[i].tensor->tensor;
  }

  for (size_t i=0 ; i<array_len(mctx->outputs); ++i) {
    outputs[i] = &mctx->outputs[i].tensor->tensor;
  }

  char* err_descr = NULL;
  torchRunModel(mctx->model->model,
                array_len(mctx->inputs), inputs,
                array_len(mctx->outputs), outputs, &err_descr);

  if (err_descr != NULL) {
    RAI_SetError(err, RAI_EMODELRUN, err_descr);
    free(err_descr);
    return 1;
  }

  for(size_t i=0 ; i<array_len(mctx->outputs) ; ++i) {
    RAI_Tensor* output_tensor = RAI_TensorCreateFromDLTensor(outputs[i]);
    mctx->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
  }

  return 0;
}

RAI_Script *RAI_ScriptCreateTorch(RAI_Device device, const char *scriptdef, RAI_Error *err) {
  size_t scriptlen = strlen(scriptdef);
  char* scriptdef_ = RedisModule_Calloc(scriptlen, sizeof(char));
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
      RAI_SetError(err, RAI_ESCRIPTCONFIGURE, "Error configuring script: unsupported device\n");
      break;
  }

  char* err_descr = NULL;
  void* script = torchCompileScript(scriptdef, dl_device, &err_descr);

  if (script == NULL) {
    RAI_SetError(err, RAI_ESCRIPTCREATE, err_descr);
    free(err_descr);
    return NULL;
  }

  RAI_Script* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->script = script;
  ret->scriptdef = scriptdef_;
  ret->device = device;
  ret->refCount = 1;

  return ret;
}

void RAI_ScriptFreeTorch(RAI_Script* script, RAI_Error* err) {

  torchDeallocContext(script->script);
  RedisModule_Free(script->scriptdef);
  RedisModule_Free(script);
}

int RAI_ScriptRunTorch(RAI_ScriptRunCtx* sctx, RAI_Error* err) {

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

  char* err_descr = NULL;
  torchRunScript(sctx->script->script, sctx->fnname, nInputs, inputs, nOutputs, outputs, &err_descr);

  if (err_descr) {
    printf("F\n");
    RAI_SetError(err, RAI_ESCRIPTRUN, err_descr);
    free(err_descr);
    return 1;
  }

  for (size_t i=0; i<nOutputs; i++) {
    sctx->outputs[0].tensor = RAI_TensorCreateFromDLTensor(outputs[i]);
  }

  return 0;
}
