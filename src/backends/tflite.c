#include "backends/tflite.h"
#include "backends/util.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "libtflite_c/tflite_c.h"

int RAI_InitBackendTFLite(int (*get_api_fn)(const char *, void *)) {
  get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
  get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
  get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
  get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
  get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));

  return REDISMODULE_OK;
}

typedef struct RAI_TfLiteBuffer {
  char* data;
  size_t len;
} RAI_TfLiteBuffer;

RAI_Model *RAI_ModelCreateTFLite(RAI_Backend backend, const char* devicestr,
                                 const char *modeldef, size_t modellen,
                                 RAI_Error *error) {
  DLDeviceType dl_device;
  
  RAI_Device device;
  int64_t deviceid;
  if (!parseDeviceStr(devicestr, &device, &deviceid)) {
    RAI_SetError(error, RAI_EMODELCONFIGURE, "Unsupported device");
    return NULL;
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
  void* model = tfliteLoadModel(modeldef, modellen, dl_device, deviceid, &error_descr, RedisModule_Alloc);

  if (model == NULL) {
    RAI_SetError(error, RAI_EMODELCREATE, error_descr);
    RedisModule_Free(error_descr);
    return NULL;
  }

  char* buffer = RedisModule_Calloc(modellen, sizeof(*buffer));
  memcpy(buffer, modeldef, modellen);

  RAI_TfLiteBuffer* tflitebuffer = RedisModule_Calloc(1, sizeof(*tflitebuffer));
  tflitebuffer->data = buffer;
  tflitebuffer->len = modellen;

  RAI_Model* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->model = model;
  ret->session = NULL;
  ret->backend = backend;
  ret->devicestr = RedisModule_Strdup(devicestr);
  ret->inputs = NULL;
  ret->outputs = NULL;
  ret->refCount = 1;
  ret->data = tflitebuffer;

  return ret;
}

void RAI_ModelFreeTFLite(RAI_Model* model, RAI_Error *error) {
  RedisModule_Free(((RAI_TfLiteBuffer*)(model->data))->data);
  RedisModule_Free(model->data);
  RedisModule_Free(model->devicestr);
  tfliteDeallocContext(model->model);

  model->model = NULL;
}

int RAI_ModelRunTFLite(RAI_ModelRunCtx* mctx, RAI_Error *error) {

  size_t ninputs = array_len(mctx->inputs);
  size_t noutputs = array_len(mctx->outputs);

  DLManagedTensor* inputs[ninputs];
  DLManagedTensor* outputs[noutputs];

  for (size_t i=0 ; i<ninputs; ++i) {
    inputs[i] = &mctx->inputs[i].tensor->tensor;
  }

  for (size_t i=0 ; i<noutputs; ++i) {
    outputs[i] = mctx->outputs[i].tensor ? &mctx->outputs[i].tensor->tensor : NULL;
  }

  char* error_descr = NULL;
  tfliteRunModel(mctx->model->model,
                 ninputs, inputs, noutputs, outputs,
                 &error_descr, RedisModule_Alloc);

  if (error_descr != NULL) {
    RAI_SetError(error, RAI_EMODELRUN, error_descr);
    RedisModule_Free(error_descr);
    return 1;
  }

  for(size_t i=0 ; i<array_len(mctx->outputs) ; ++i) {
    if (outputs[i] == NULL) {
      RAI_SetError(error, RAI_EMODELRUN, "Model did not generate the expected number of outputs.");
      return 1;
    }
    RAI_Tensor* output_tensor = RAI_TensorCreateFromDLTensor(outputs[i]);
    mctx->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
    RAI_TensorFree(output_tensor);
    RedisModule_Free(outputs[i]);
  }

  return 0;
}

int RAI_ModelSerializeTFLite(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
  RAI_TfLiteBuffer* tflitebuffer = (RAI_TfLiteBuffer*)model->data;
  *buffer = RedisModule_Calloc(tflitebuffer->len, sizeof(char));
  memcpy(*buffer, tflitebuffer->data, tflitebuffer->len);
  *len = tflitebuffer->len;

  return 0;
}
