/**
 * model.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the RedisModuleType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#include "model.h"
#include "model_struct.h"
#include "backends.h"
#include "stats.h"
#include "backends/util.h"
#include <pthread.h>
#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "run_info.h"

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
    .minbatchsize = minbatchsize,
    .backends_intra_op_parallelism = getBackendsIntraOpParallelism(),
    .backends_inter_op_parallelism = getBackendsInterOpParallelism(),
  };

  size_t len;
  char *buffer = NULL;

  if (encver <= 100) {
    buffer = RedisModule_LoadStringBuffer(io, &len);
  }
  else {
    len = RedisModule_LoadUnsigned(io);
    buffer = RedisModule_Alloc(len);
    const size_t n_chunks = RedisModule_LoadUnsigned(io);
    long long chunk_offset = 0;
    for (size_t i=0; i<n_chunks; i++) {
      size_t chunk_len;
      char *chunk_buffer = RedisModule_LoadStringBuffer(io, &chunk_len);
      memcpy(buffer + chunk_offset, chunk_buffer, chunk_len);
      chunk_offset += chunk_len;
      RedisModule_Free(chunk_buffer);
    }
  }

  RAI_Error err = {0};

  RAI_Model *model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                                     buffer, len, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModuleCtx* ctx = RedisModule_GetContextFromIO(io);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load default backend");
      RAI_ClearError(&err);
      return NULL;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, buffer, len, &err);
  }
 
  if (err.code != RAI_OK) {
    RedisModuleCtx* ctx = RedisModule_GetContextFromIO(io);
    RedisModule_Log(ctx, "error", err.detail);
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
    RedisModuleCtx* stats_ctx = RedisModule_GetContextFromIO(io);
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
  long long chunk_size = getModelChunkSize();
  const size_t n_chunks = len / chunk_size + 1;
  RedisModule_SaveUnsigned(io, len);
  RedisModule_SaveUnsigned(io, n_chunks);
  for (size_t i=0; i<n_chunks; i++) {
    size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
    RedisModule_SaveStringBuffer(io, buffer + i * chunk_size, chunk_len);
  }

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

  long long chunk_size = getModelChunkSize();
  const size_t n_chunks = len / chunk_size + 1;
  RedisModuleString **buffers_ = array_new(RedisModuleString*, n_chunks);

  for (size_t i=0; i<n_chunks; i++) {
    size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
    array_append(buffers_, RedisModule_CreateString(ctx, buffer + i * chunk_size, chunk_len));
  }

  if (buffer) {
    RedisModule_Free(buffer);
  }

  const char* backendstr = RAI_BackendName(model->backend);

  RedisModule_EmitAOF(aof, "AI.MODELSET", "slccclclcvcvcv",
                      key,
                      backendstr, model->devicestr, model->tag,
                      "BATCHSIZE", model->opts.batchsize,
                      "MINBATCHSIZE", model->opts.minbatchsize,
                      "INPUTS", inputs_, model->ninputs,
                      "OUTPUTS", outputs_, model->noutputs,
                      "BLOB", buffers_, n_chunks);

  for (size_t i=0; i<model->ninputs; i++) {
    RedisModule_FreeString(ctx, inputs_[i]);
  }
  array_free(inputs_);

  for (size_t i=0; i<model->noutputs; i++) {
    RedisModule_FreeString(ctx, outputs_[i]);
  }
  array_free(outputs_);

  for (size_t i=0; i<n_chunks; i++) {
    RedisModule_FreeString(ctx, buffers_[i]);
  }
  array_free(buffers_);
}


/* Return REDISMODULE_ERR if there was an error getting the Model.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key, RAI_Model **model,
                              int mode) {
  *key = RedisModule_OpenKey(ctx, keyName, mode);
  if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, "ERR model key is empty");
    return REDISMODULE_ERR;
  }
  if (RedisModule_ModuleTypeGetType(*key) != RedisAI_ModelType) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    return REDISMODULE_ERR;
  }
  *model = RedisModule_ModuleTypeGetValue(*key);
  return REDISMODULE_OK;
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

  RedisAI_ModelType = RedisModule_CreateDataType(ctx, "AI__MODEL", RAI_ENC_VER_MM, &tmModel);
  return RedisAI_ModelType != NULL;
}

RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char* devicestr, const char* tag, RAI_ModelOpts opts,
                           size_t ninputs, const char **inputs,
                           size_t noutputs, const char **outputs,
                           const char *modeldef, size_t modellen, RAI_Error* err) {
  RAI_Model *model;
  if (backend == RAI_BACKEND_TENSORFLOW) {
    if (!RAI_backends.tf.model_create_with_nodes) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
      return NULL;
    }
    model = RAI_backends.tf.model_create_with_nodes(backend, devicestr, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_TFLITE) {
    if (!RAI_backends.tflite.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
      return NULL;
    }
    model = RAI_backends.tflite.model_create(backend, devicestr, opts, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_TORCH) {
    if (!RAI_backends.torch.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
      return NULL;
    }
    model = RAI_backends.torch.model_create(backend, devicestr, opts, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_ONNXRUNTIME) {
    if (!RAI_backends.onnx.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
      return NULL;
    }
    model = RAI_backends.onnx.model_create(backend, devicestr, opts, modeldef, modellen, err);
  }
  else {
    RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "ERR Unsupported backend");
    return NULL;
  }

  if (model) {
    model->tag = RedisModule_Strdup(tag);
  }

  return model;
}

void RAI_ModelFree(RAI_Model* model, RAI_Error* err) {
  if (__atomic_sub_fetch(&model->refCount, 1, __ATOMIC_RELAXED) > 0){
    return;
  }

  if (model->backend == RAI_BACKEND_TENSORFLOW) {
    if (!RAI_backends.tf.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
      return;
    }
    RAI_backends.tf.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_TFLITE) {
    if (!RAI_backends.tflite.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
      return;
    }
    RAI_backends.tflite.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_TORCH) {
    if (!RAI_backends.torch.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
      return;
    }
    RAI_backends.torch.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_ONNXRUNTIME) {
    if (!RAI_backends.onnx.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
      return;
    }
    RAI_backends.onnx.model_free(model, err);
  }
  else {
    RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend");
    return;
  }

  RedisModule_Free(model->tag);

  RAI_RemoveStatsEntry(model->infokey);

  RedisModule_Free(model);
}

RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model) {
#define PARAM_INITIAL_SIZE 10
  RAI_ModelRunCtx* mctx = RedisModule_Calloc(1, sizeof(*mctx));
  mctx->model = RAI_ModelGetShallowCopy(model);
  mctx->inputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
  mctx->outputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
  return mctx;
#undef PARAM_INITIAL_SIZE
}

static int Model_RunCtxAddParam(RAI_ModelRunCtx* mctx, RAI_ModelCtxParam** paramArr,
                                const char* name, RAI_Tensor* tensor) {

  RAI_ModelCtxParam param = {
      .name = name,
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  *paramArr = array_append(*paramArr, param);
  return 1;
}

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* mctx, const char* inputName, RAI_Tensor* inputTensor) {
  return Model_RunCtxAddParam(mctx, &mctx->inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* mctx, const char* outputName) {
  return Model_RunCtxAddParam(mctx, &mctx->outputs, outputName, NULL);
}

size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx* mctx) {
  return array_len(mctx->inputs);
}

size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* mctx) {
  return array_len(mctx->outputs);
}

RAI_Tensor* RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx* mctx, size_t index) {
  assert(RAI_ModelRunCtxNumInputs(mctx) > index && index >= 0);
  return mctx->inputs[index].tensor;
}

RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* mctx, size_t index) {
  assert(RAI_ModelRunCtxNumOutputs(mctx) > index && index >= 0);
  return mctx->outputs[index].tensor;
}

void RAI_ModelRunCtxFree(RAI_ModelRunCtx* mctx, int freeTensors) {
  if (freeTensors) {
    for (size_t i=0; i<array_len(mctx->inputs); ++i) {
      RAI_TensorFree(mctx->inputs[i].tensor);
    }

    for (size_t i = 0 ; i < array_len(mctx->outputs) ; ++i) {
      if (mctx->outputs[i].tensor) {
        RAI_TensorFree(mctx->outputs[i].tensor);
      }
    }
  }

  array_free(mctx->inputs);
  array_free(mctx->outputs);

  RAI_Error err = {0};
  RAI_ModelFree(mctx->model, &err);

  if (err.code != RAI_OK) {
    // TODO: take it to client somehow
    RAI_ClearError(&err);
  }

  RedisModule_Free(mctx);
}

int RAI_ModelRun(RAI_ModelRunCtx** mctxs, long long n, RAI_Error* err) {
  int ret;

  if (n == 0) {
    RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Nothing to run");
    return REDISMODULE_ERR;
  }

  RAI_ModelRunCtx** mctxs_arr = array_newlen(RAI_ModelRunCtx*, n);
  for (int i=0; i<n; i++) {
    mctxs_arr[i] = mctxs[i];
  }

  switch (mctxs_arr[0]->model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      if (!RAI_backends.tf.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tf.model_run(mctxs_arr, err);
      break;
    case RAI_BACKEND_TFLITE:
      if (!RAI_backends.tflite.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tflite.model_run(mctxs_arr, err);
      break;
    case RAI_BACKEND_TORCH:
      if (!RAI_backends.torch.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.torch.model_run(mctxs_arr, err);
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      if (!RAI_backends.onnx.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.onnx.model_run(mctxs_arr, err);
      break;
    default:
      RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "ERR Unsupported backend");
      return REDISMODULE_ERR;
  }

  array_free(mctxs_arr);

  return ret;
}

RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model) {
  __atomic_fetch_add(&model->refCount, 1, __ATOMIC_RELAXED);  
  return model;
}

int RAI_ModelSerialize(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err) {
  int ret;

  switch (model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      if (!RAI_backends.tf.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tf.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_TFLITE:
      if (!RAI_backends.tflite.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tflite.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_TORCH:
      if (!RAI_backends.torch.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.torch.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      if (!RAI_backends.onnx.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.onnx.model_serialize(model, buffer, len, err);
      break;
    default:
      RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "ERR Unsupported backend");
      return REDISMODULE_ERR;
  }

  return ret;
}


int RedisAI_Parse_ModelRun_RedisCommand(RedisModuleCtx *ctx,
                                        RedisModuleString **argv, int argc,
                                        RAI_ModelRunCtx **mctx,
                                        RedisModuleString ***inkeys,
                                        RedisModuleString ***outkeys,
                                        RAI_Model **mto,
                                        RAI_Error *error) {
  if (argc < 3) {
    RAI_SetError(error, RAI_EMODELRUN,
                 "ERR wrong number of arguments for 'AI.MODELRUN' command");
    return -1;
  }

  const char *inputstr = RedisModule_StringPtrLen(argv[2], NULL);
  if (strcasecmp(inputstr, "INPUTS")) {
    RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
    return -1;
  }

  int is_input = 0;
  size_t ninputs = 0;
  size_t noutputs = 0;
  int outputs_flag_count = 0;
  size_t argpos = 3;

  for (; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "OUTPUTS") && outputs_flag_count == 0) {
      is_input = 1;
      outputs_flag_count = 1;
    } else {
      RedisModule_RetainString(ctx, argv[argpos]);
      if (is_input == 0) {
        *inkeys = array_append(*inkeys, argv[argpos]);
        ninputs++;
      } else {
        *outkeys = array_append(*outkeys, argv[argpos]);
        noutputs++;
      }
    }
  }

  if ((*mto)->inputs && array_len((*mto)->inputs) != ninputs) {
    RAI_SetError(
        error, RAI_EMODELRUN,
        "Number of names given as INPUTS during MODELSET and keys given as "
        "INPUTS here do not match");
    return -1;
  }

  if ((*mto)->outputs && array_len((*mto)->outputs) != noutputs) {
    RAI_SetError(
        error, RAI_EMODELRUN,
        "Number of names given as OUTPUTS during MODELSET and keys given as "
        "INPUTS here do not match");
    return -1;
  }
  return argpos;
}

RedisModuleType *RAI_ModelRedisType(void) {
    return RedisAI_ModelType;
}
