#include "decode_v0.h"
#include "assert.h"

void* RAI_RDBLoadTensor_v0(RedisModuleIO *io) {
  DLContext ctx;
  ctx.device_type = RedisModule_LoadUnsigned(io);
  
  ctx.device_id = RedisModule_LoadUnsigned(io);

  // For now we only support CPU tensors (except during model and script run)
  assert(ctx.device_type == kDLCPU);
  assert(ctx.device_id == 0);

  DLDataType dtype;
  dtype.bits = RedisModule_LoadUnsigned(io);
  dtype.code = RedisModule_LoadUnsigned(io);
  dtype.lanes = RedisModule_LoadUnsigned(io);

  size_t ndims = RedisModule_LoadUnsigned(io);

  int64_t* shape = RedisModule_Calloc(ndims, sizeof(*shape));
  int64_t* strides = RedisModule_Calloc(ndims, sizeof(*strides));
  for (size_t i = 0 ; i < ndims ; ++i){
    shape[i] = RedisModule_LoadUnsigned(io);
  }

  for (size_t i = 0 ; i < ndims ; ++i){
    strides[i] = RedisModule_LoadUnsigned(io);
  }

  size_t byte_offset = RedisModule_LoadUnsigned(io);
  
  size_t len;
  char *data = RedisModule_LoadStringBuffer(io, &len);

  RAI_Tensor *ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->tensor = (DLManagedTensor){
    .dl_tensor = (DLTensor){
      .ctx = ctx,
      .data = data,
      .ndim = ndims,
      .dtype = dtype,
      .shape = shape,
      .strides = strides,
      .byte_offset = 0
    },
    .manager_ctx = NULL,
    .deleter = NULL
  };
  ret->refCount = 1;
  return ret;
}

void* RAI_RDBLoadModel_v0(RedisModuleIO *io) {
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
  char *buffer = RedisModule_LoadStringBuffer(io, &len);

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
    RedisModule_Log(ctx, "error", "%s", err.detail);
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

void* RAI_RDBLoadScript_v0(RedisModuleIO *io) {
  RAI_Error err = {0};

  const char* devicestr = RedisModule_LoadStringBuffer(io, NULL);
  const char* tag = RedisModule_LoadStringBuffer(io, NULL);

  size_t len;
  char* scriptdef = RedisModule_LoadStringBuffer(io, &len);

  RAI_Script* script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModuleCtx* ctx = RedisModule_GetContextFromIO(io);
    int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load default TORCH backend\n");
      RAI_ClearError(&err);
      return NULL;
    }
    RAI_ClearError(&err);
    script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);
  }

  RedisModule_Free(scriptdef);

  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }

  RedisModuleCtx* stats_ctx = RedisModule_GetContextFromIO(io);
  RedisModuleString* stats_keystr = RedisModule_CreateStringFromString(
      stats_ctx, RedisModule_GetKeyNameFromIO(io));
  const char* stats_devicestr = RedisModule_Strdup(devicestr);
  const char* stats_tag = RedisModule_Strdup(tag);

  script->infokey =
      RAI_AddStatsEntry(stats_ctx, stats_keystr, RAI_SCRIPT, RAI_BACKEND_TORCH,
                        stats_devicestr, stats_tag);

  RedisModule_Free(stats_keystr);

  return script;
}