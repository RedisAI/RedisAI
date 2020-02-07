#include "backends/torch.h"
#include "backends/util.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "libtorch_c/torch_c.h"

int RAI_InitBackendTorch(int (*get_api_fn)(const char *, void *)) {
  get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
  get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
  get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
  get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
  get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));

  return REDISMODULE_OK;
}

RAI_Model *RAI_ModelCreateTorch(RAI_Backend backend, const char* devicestr, RAI_ModelOpts opts,
                                const char *modeldef, size_t modellen,
                                RAI_Error *error) {
  DLDeviceType dl_device;
  
  RAI_Device device;
  int64_t deviceid;

  if (!parseDeviceStr(devicestr, &device, &deviceid)) {
    RAI_SetError(error, RAI_EMODELCONFIGURE, "ERR unsupported device");
  }

  switch (device) {
    case RAI_DEVICE_CPU:
      dl_device = kDLCPU;
      break;
    case RAI_DEVICE_GPU:
      dl_device = kDLGPU;
      break;
    default:
      RAI_SetError(error, RAI_EMODELCONFIGURE, "Error configuring model: unsupported device\n");
      return NULL;
  }

  char* error_descr = NULL;
  void* model = torchLoadModel(modeldef, modellen, dl_device, deviceid, &error_descr, RedisModule_Alloc);

  if (model == NULL) {
    RAI_SetError(error, RAI_EMODELCREATE, error_descr);
    RedisModule_Free(error_descr);
    return NULL;
  }

  RAI_Model* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->model = model;
  ret->session = NULL;
  ret->backend = backend;
  ret->devicestr = RedisModule_Strdup(devicestr);
  ret->inputs = NULL;
  ret->outputs = NULL;
  ret->opts = opts;
  ret->refCount = 1;

  return ret;
}

void RAI_ModelFreeTorch(RAI_Model* model, RAI_Error *error) {
  torchDeallocContext(model->model);
}

int RAI_ModelRunTorch(RAI_ModelRunCtx* mctx, RAI_Error *error) {
  const size_t nbatches = array_len(mctx->batches);
  if (nbatches == 0) {
    RAI_SetError(error, RAI_EMODELRUN, "No batches to run\n");
    return 1;
  }

  size_t ninputs = array_len(mctx->batches[0].inputs);
  size_t noutputs = array_len(mctx->batches[0].outputs);

  RAI_Tensor* inputs[ninputs];

  DLManagedTensor* inputs_dl[ninputs];
  DLManagedTensor* outputs_dl[noutputs];

  size_t batch_sizes[nbatches];
  size_t batch_offsets[nbatches];

  if (nbatches > 1) {
    size_t total_batch_size = 0;
    if (array_len(mctx->batches[0].inputs) > 0) {
      for (size_t b=0; b<nbatches; ++b) {
        batch_sizes[b] = RAI_TensorDim(mctx->batches[b].inputs[0].tensor, 0);
        total_batch_size += batch_sizes[b];
      }
      batch_offsets[0] = 0;
      for (size_t b=1; b<nbatches; ++b) {
        batch_offsets[b] = batch_sizes[b-1];
      }
    }

    for (size_t i=0 ; i<ninputs; ++i) {
      RAI_Tensor* batch[nbatches];

      for (size_t b=0; b<nbatches; b++) {
        batch[b] = mctx->batches[b].inputs[i].tensor;
      }

      inputs[i] = RAI_TensorCreateByConcatenatingTensors(batch, nbatches);
      inputs_dl[i] = &inputs[i]->tensor;
    }
  }
  else {
    for (size_t i=0 ; i<ninputs; ++i) {
      inputs[i] = RAI_TensorGetShallowCopy(mctx->batches[0].inputs[i].tensor);
      inputs_dl[i] = &inputs[i]->tensor;
    }
  }

  for (size_t i=0 ; i<noutputs; ++i) {
    outputs_dl[i] = NULL;
  }

  char* error_descr = NULL;
  torchRunModel(mctx->model->model,
                ninputs, inputs_dl, noutputs, outputs_dl,
                &error_descr, RedisModule_Alloc);

  if (error_descr != NULL) {
    RAI_SetError(error, RAI_EMODELRUN, error_descr);
    RedisModule_Free(error_descr);
    return 1;
  }

  for(size_t i=0 ; i<noutputs; ++i) {
    if (outputs_dl[i] == NULL) {
      RAI_SetError(error, RAI_EMODELRUN, "Model did not generate the expected number of outputs.");
      return 1;
    }
    RAI_Tensor* output_tensor = RAI_TensorCreateFromDLTensor(outputs_dl[i]);
    if (nbatches > 1) {
      for (size_t b=0; b<nbatches; b++) {
        mctx->batches[b].outputs[i].tensor = RAI_TensorCreateBySlicingTensor(output_tensor, batch_offsets[b], batch_sizes[b]);
      }
    }
    else {
      mctx->batches[0].outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
    }
    RAI_TensorFree(output_tensor);
  }

  for (size_t i=0 ; i<ninputs; ++i) {
    RAI_TensorFree(inputs[i]);
  }

  return 0;
}

int RAI_ModelSerializeTorch(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
  char* error_descr = NULL;
  torchSerializeModel(model->model, buffer, len, &error_descr, RedisModule_Alloc);

  if (*buffer == NULL) {
    RAI_SetError(error, RAI_EMODELSERIALIZE, error_descr);
    RedisModule_Free(error_descr);
    return 1;
  }

  return 0;
}

RAI_Script *RAI_ScriptCreateTorch(const char* devicestr, const char *scriptdef, RAI_Error *error) {
  DLDeviceType dl_device;
  
  RAI_Device device;
  int64_t deviceid;

  if (!parseDeviceStr(devicestr, &device, &deviceid)) {
    RAI_SetError(error, RAI_ESCRIPTCONFIGURE, "ERR unsupported device");
  }


  switch (device) {
    case RAI_DEVICE_CPU:
      dl_device = kDLCPU;
      break;
    case RAI_DEVICE_GPU:
      dl_device = kDLGPU;
      break;
    default:
      RAI_SetError(error, RAI_ESCRIPTCONFIGURE, "Error configuring script: unsupported device\n");
      break;
  }

  char* error_descr = NULL;
  void* script = torchCompileScript(scriptdef, dl_device, deviceid, &error_descr, RedisModule_Alloc);

  if (script == NULL) {
    RAI_SetError(error, RAI_ESCRIPTCREATE, error_descr);
    RedisModule_Free(error_descr);
    return NULL;
  }

  RAI_Script* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->script = script;
  ret->scriptdef = RedisModule_Strdup(scriptdef);
  ret->devicestr = RedisModule_Strdup(devicestr);
  ret->refCount = 1;
  

  return ret;
}

void RAI_ScriptFreeTorch(RAI_Script* script, RAI_Error* error) {

  torchDeallocContext(script->script);
  RedisModule_Free(script->scriptdef);
  RedisModule_Free(script);
}

int RAI_ScriptRunTorch(RAI_ScriptRunCtx* sctx, RAI_Error* error) {

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

  char* error_descr = NULL;
  torchRunScript(sctx->script->script, sctx->fnname,
                 nInputs, inputs, nOutputs, outputs,
                 &error_descr, RedisModule_Alloc);

  if (error_descr) {
    RAI_SetError(error, RAI_ESCRIPTRUN, error_descr);
    RedisModule_Free(error_descr);
    return 1;
  }

  for (size_t i=0; i<nOutputs; i++) {
    sctx->outputs[i].tensor = RAI_TensorCreateFromDLTensor(outputs[i]);
  }

  return 0;
}
