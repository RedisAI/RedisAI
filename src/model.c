#include "model.h"
#include "model_struct.h"
#include "backends.h"
#include "stats.h"

#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"

RedisModuleType *RedisAI_ModelType = NULL;

static void* RAI_Model_RdbLoad(struct RedisModuleIO *io, int encver) {
  // if (encver != RAI_ENC_VER) {
  //   /* We should actually log an error here, or try to implement
  //      the ability to load older versions of our data structure. */
  //   return NULL;
  // }

  RAI_Backend backend = RedisModule_LoadUnsigned(io);
  const char *devicestr = RedisModule_LoadStringBuffer(io, NULL);

  const char *tag = RedisModule_LoadStringBuffer(io, NULL);

  const size_t batchsize = RedisModule_LoadUnsigned(io);
  const size_t minbatchsize = RedisModule_LoadUnsigned(io);

  const size_t ninputs = RedisModule_LoadUnsigned(io);
  const char **inputs = RedisModule_Alloc(ninputs * sizeof(char*));

  for (size_t i=0; i<ninputs; i++) {
    inputs[i] = RedisModule_LoadStringBuffer(io, NULL);
  }

  const size_t noutputs = RedisModule_LoadUnsigned(io);

  const char **outputs = RedisModule_Alloc(ninputs * sizeof(char*));

  for (size_t i=0; i<noutputs; i++) {
    outputs[i] = RedisModule_LoadStringBuffer(io, NULL);
  }

  RAI_ModelOpts opts = {
    .batchsize = batchsize,
    .minbatchsize = minbatchsize
  };

  size_t len;

  char *buffer = RedisModule_LoadStringBuffer(io, &len);

  RAI_Error err = {0};

  RAI_Model *model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                                     buffer, len, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModuleCtx* ctx = RedisModule_GetContextFromIO(io);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load default backend\n");
      RAI_ClearError(&err);
      return NULL;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, buffer, len, &err);
  }
 
  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return NULL;
  }

  RedisModule_Free(inputs);
  RedisModule_Free(outputs);
  RedisModule_Free(buffer);

  RedisModuleCtx* stats_ctx = RedisModule_GetContextFromIO(io);
  RedisModuleString* stats_keystr = RedisModule_CreateStringFromString(stats_ctx,
                                                                       RedisModule_GetKeyNameFromIO(io));
  const char* stats_devicestr = RedisModule_Strdup(devicestr);
  const char* stats_tag = RedisModule_Strdup(tag);

  model->infokey = RAI_AddStatsEntry(stats_ctx, stats_keystr, RAI_MODEL, backend, stats_devicestr, stats_tag);

  RedisModule_Free(stats_keystr);

  return model;
}

static void RAI_Model_RdbSave(RedisModuleIO *io, void *value) {
  RAI_Model *model = (RAI_Model*)value;
  char *buffer = NULL;
  size_t len = 0;
  RAI_Error err = {0};

  int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return;
  }

  RedisModule_SaveUnsigned(io, model->backend);
  RedisModule_SaveStringBuffer(io, model->devicestr, strlen(model->devicestr) + 1);
  RedisModule_SaveStringBuffer(io, model->tag, strlen(model->tag) + 1);
  RedisModule_SaveUnsigned(io, model->opts.batchsize);
  RedisModule_SaveUnsigned(io, model->opts.minbatchsize);
  RedisModule_SaveUnsigned(io, model->ninputs);
  for (size_t i=0; i<model->ninputs; i++) {
    RedisModule_SaveStringBuffer(io, model->inputs[i], strlen(model->inputs[i]) + 1);
  }
  RedisModule_SaveUnsigned(io, model->noutputs);
  for (size_t i=0; i<model->noutputs; i++) {
    RedisModule_SaveStringBuffer(io, model->outputs[i], strlen(model->outputs[i]) + 1);
  }
  RedisModule_SaveStringBuffer(io, buffer, len);

  if (buffer) {
    RedisModule_Free(buffer);
  }
}

static void RAI_Model_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  RAI_Model *model = (RAI_Model*)value;

  char *buffer = NULL;
  size_t len = 0;
  RAI_Error err = {0};

  int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return;
  }

  // AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob

  RedisModuleString **inputs_ = array_new(RedisModuleString*, model->ninputs);
  RedisModuleString **outputs_ = array_new(RedisModuleString*, model->noutputs);

  RedisModuleCtx *ctx = RedisModule_GetContextFromIO(aof);

  for (size_t i=0; i<model->ninputs; i++) {
    array_append(inputs_, RedisModule_CreateString(ctx, model->inputs[i], strlen(model->inputs[i])));
  }

  for (size_t i=0; i<model->noutputs; i++) {
    array_append(outputs_, RedisModule_CreateString(ctx, model->outputs[i], strlen(model->outputs[i])));
  }

  const char* backendstr = RAI_BackendName(model->backend);

  RedisModule_EmitAOF(aof, "AI.MODELSET", "slccclclcvcvb",
                      key,
                      backendstr, model->devicestr, model->tag,
                      "BATCHSIZE", model->opts.batchsize,
                      "MINBATCHSIZE", model->opts.minbatchsize,
                      "INPUTS", inputs_, model->ninputs,
                      "OUTPUTS", outputs_, model->noutputs,
                      buffer, len);

  if (buffer) {
    RedisModule_Free(buffer);
  }

  for (size_t i=0; i<model->ninputs; i++) {
    RedisModule_FreeString(ctx, inputs_[i]);
  }

  array_free(inputs_);

  for (size_t i=0; i<model->noutputs; i++) {
    RedisModule_FreeString(ctx, outputs_[i]);
  }

  array_free(outputs_);
}

// TODO: pass err in?
static void RAI_Model_DTFree(void *value) {
  RAI_Error err = {0};
  RAI_ModelFree(value, &err);
  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }
}

int RAI_ModelInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmModel = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RAI_Model_RdbLoad,
      .rdb_save = RAI_Model_RdbSave,
      .aof_rewrite = RAI_Model_AofRewrite,
      .mem_usage = NULL,
      .free = RAI_Model_DTFree,
      .digest = NULL
  };

  RedisAI_ModelType = RedisModule_CreateDataType(ctx, "AI__MODEL", 0, &tmModel);
  return RedisAI_ModelType != NULL;
}

RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char* devicestr, const char* tag, RAI_ModelOpts opts,
                           size_t ninputs, const char **inputs,
                           size_t noutputs, const char **outputs,
                           const char *modeldef, size_t modellen, RAI_Error* err) {
  RAI_Model *model;
  if (backend == RAI_BACKEND_TENSORFLOW) {
    if (!RAI_backends.tf.model_create_with_nodes) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
      return NULL;
    }
    model = RAI_backends.tf.model_create_with_nodes(backend, devicestr, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_TFLITE) {
    if (!RAI_backends.tflite.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TFLITE.\n");
      return NULL;
    }
    model = RAI_backends.tflite.model_create(backend, devicestr, opts, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_TORCH) {
    if (!RAI_backends.torch.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
      return NULL;
    }
    model = RAI_backends.torch.model_create(backend, devicestr, opts, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_ONNXRUNTIME) {
    if (!RAI_backends.onnx.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
      return NULL;
    }
    model = RAI_backends.onnx.model_create(backend, devicestr, opts, modeldef, modellen, err);
  }
  else {
    RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend.\n");
    return NULL;
  }

  if (model) {
    model->tag = RedisModule_Strdup(tag);
  }

  return model;
}

void RAI_ModelFree(RAI_Model* model, RAI_Error* err) {
  if (--model->refCount > 0){
    return;
  }

  if (model->backend == RAI_BACKEND_TENSORFLOW) {
    if (!RAI_backends.tf.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
      return;
    }
    RAI_backends.tf.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_TFLITE) {
    if (!RAI_backends.tflite.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TFLITE.\n");
      return;
    }
    RAI_backends.tflite.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_TORCH) {
    if (!RAI_backends.torch.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
      return;
    }
    RAI_backends.torch.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_ONNXRUNTIME) {
    if (!RAI_backends.onnx.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
      return;
    }
    RAI_backends.onnx.model_free(model, err);
  }
  else {
    RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend\n");
    return;
  }

  RedisModule_Free(model->tag);

  RAI_RemoveStatsEntry(model->infokey);

  RedisModule_Free(model);
}

RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model) {
#define BATCH_INITIAL_SIZE 10
  RAI_ModelRunCtx* mctx = RedisModule_Calloc(1, sizeof(*mctx));
  mctx->model = RAI_ModelGetShallowCopy(model);
  mctx->nbatches=0;
  mctx->batches = array_new(RAI_ModelCtxBatch, BATCH_INITIAL_SIZE);
#undef BATCH_INITIAL_SIZE
  return mctx;
}

static int Model_RunCtxAddParam(RAI_ModelRunCtx* mctx, RAI_ModelCtxParam** paramArr, 
                                const char* name, RAI_Tensor* tensor) {

  RAI_ModelCtxParam param = {
      .name = name,
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  *paramArr = array_append(*paramArr, param);
  return REDISMODULE_OK;
}

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* mctx, size_t id, const char* inputName, RAI_Tensor* inputTensor) {
  if (id >= RAI_ModelRunCtxNumBatches(mctx)) {
    // TODO error
    return REDISMODULE_ERR;
  }
  mctx->batches[id].ninputs++;
  return Model_RunCtxAddParam(mctx, &mctx->batches[id].inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* mctx, size_t id, const char* outputName) {
  if (id >= RAI_ModelRunCtxNumBatches(mctx)) {
    // TODO error
    return REDISMODULE_ERR;
  }
  mctx->batches[id].noutputs++;
  return Model_RunCtxAddParam(mctx, &mctx->batches[id].outputs, outputName, NULL);
}

size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx* mctx) {
  if (RAI_ModelRunCtxNumBatches(mctx) == 0) {
    return 0;
  }
  // Here we assume batch is well-formed (i.e. number of outputs is equal in all batches)
  return mctx->batches[0].ninputs;
}

size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* mctx) {
  if (RAI_ModelRunCtxNumBatches(mctx)) {
    return 0;
  }
  // Here we assume batch is well-formed (i.e. number of outputs is equal in all batches)
  return mctx->batches[0].noutputs;
}

int RAI_ModelRunCtxAddBatch(RAI_ModelRunCtx* mctx) {
#define PARAM_INITIAL_SIZE 10
  RAI_ModelCtxBatch batch = {
    .inputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE),
    .ninputs = 0,
    .outputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE),
    .noutputs = 0
  };
#undef PARAM_INITIAL_SIZE
  array_append(mctx->batches, batch);
  mctx->batches++;
  return array_len(mctx->batches)-1;
}

size_t RAI_ModelRunCtxNumBatches(RAI_ModelRunCtx* mctx) {
  return array_len(mctx->batches);
}

void RAI_ModelRunCtxCopyBatch(RAI_ModelRunCtx* dest, size_t id_dest, RAI_ModelRunCtx* src, size_t id_src) {
  const size_t ninputs = src->batches[id_src].ninputs;
  const size_t noutputs = src->batches[id_src].noutputs;

  for (size_t i=0; i<ninputs; i++) {
    RAI_ModelCtxParam param = src->batches[id_src].inputs[i];
    RAI_ModelRunCtxAddInput(dest, id_dest, param.name, param.tensor);
  }

  for (size_t i=0; i<noutputs; i++) {
    RAI_ModelCtxParam param = src->batches[id_src].outputs[i];
    RAI_ModelRunCtxAddOutput(dest, id_dest, param.name);
  }
}

RAI_Tensor* RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx* mctx, size_t id, size_t index) {
  // TODO: add method to collect from batches?
  assert(RAI_ModelRunCtxNumInputs(mctx) > index && index >= 0);
  return mctx->batches[id].inputs[index].tensor;
}

RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* mctx, size_t id, size_t index) {
  // TODO: add method to collect from batches?
  assert(RAI_ModelRunCtxNumOutputs(mctx) > index && index >= 0);
  return mctx->batches[id].outputs[index].tensor;
}

void RAI_ModelRunCtxFree(RAI_ModelRunCtx* mctx) {
  for (size_t b=0; b<array_len(mctx->batches); ++b) {
    for (size_t i=0; i<array_len(mctx->batches[b].inputs); ++i) {
      RAI_TensorFree(mctx->batches[b].inputs[i].tensor);
    }
    array_free(mctx->batches[b].inputs);

    for (size_t i = 0 ; i < array_len(mctx->batches[b].outputs) ; ++i) {
      if (mctx->batches[b].outputs[i].tensor) {
        RAI_TensorFree(mctx->batches[b].outputs[i].tensor);
      }
    }
    array_free(mctx->batches[b].outputs);
  }

  RAI_Error err = {0};
  RAI_ModelFree(mctx->model, &err);

  if (err.code != RAI_OK) {
    // TODO: take it to client somehow
    RAI_ClearError(&err);
  }

  RedisModule_Free(mctx);
}

int RAI_ModelRun(RAI_ModelRunCtx* mctx, RAI_Error* err) {
  int ret = REDISMODULE_ERR;

  switch (mctx->model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      if (!RAI_backends.tf.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tf.model_run(mctx, err);
      break;
    case RAI_BACKEND_TFLITE:
      if (!RAI_backends.tflite.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TFLITE.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tflite.model_run(mctx, err);
      break;
    case RAI_BACKEND_TORCH:
      if (!RAI_backends.torch.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.torch.model_run(mctx, err);
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      if (!RAI_backends.onnx.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.onnx.model_run(mctx, err);
      break;
    default:
      RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend.\n");
      return REDISMODULE_ERR;
  }

  return ret;
}

RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model) {
  ++model->refCount;
  return model;
}

int RAI_ModelSerialize(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err) {
  int ret;

  switch (model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      if (!RAI_backends.tf.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tf.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_TFLITE:
      if (!RAI_backends.tflite.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TFLITE.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tflite.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_TORCH:
      if (!RAI_backends.torch.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.torch.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      if (!RAI_backends.onnx.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.onnx.model_serialize(model, buffer, len, err);
      break;
    default:
      RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend.\n");
      return REDISMODULE_ERR;
  }

  return ret;
}
