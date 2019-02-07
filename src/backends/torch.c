#include "backends/torch.h"
#include "tensor.h"
#include "utils/arr_rm_alloc.h"
#include "torch_c.h"


RAI_Graph *RAI_GraphCreateTorch(RAI_Backend backend, RAI_Device device,
                                const char *graphdef, size_t graphlen) {
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

  void* graph = torchLoadGraph(graphdef, dl_device);

  RAI_Graph* ret = RedisModule_Alloc(sizeof(*ret));
  ret->graph = graph;
  ret->session = NULL;
  ret->backend = backend;
  ret->refCount = 1;

  return ret;
}

void RAI_GraphFreeTorch(RAI_Graph* graph) {
  torchDeallocContext(graph->graph);
}

int RAI_GraphRunTorch(RAI_GraphRunCtx* gctx) {

  DLManagedTensor** inputs = RedisModule_Alloc(sizeof(*inputs));
  DLManagedTensor** outputs = RedisModule_Alloc(sizeof(*outputs));

  for (size_t i=0 ; i<array_len(gctx->inputs); ++i) {
    inputs[i] = &gctx->inputs[i].tensor->tensor;
  }

  for (size_t i=0 ; i<array_len(gctx->outputs); ++i) {
    outputs[i] = &gctx->outputs[i].tensor->tensor;
  }

  long ret = torchRunGraph(gctx->graph,
                           array_len(gctx->inputs), inputs,
                           array_len(gctx->outputs), outputs);

  return 0;
}

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

  torchDeallocContext(script->script);
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
