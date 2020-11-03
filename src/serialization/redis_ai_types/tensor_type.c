#include "tensor_type.h"
#include "../AOF/rai_aof_rewrite.h"
#include "../RDB/encoder/rai_rdb_encode.h"


RedisModuleType *RedisAI_TensorType = NULL;

static void RAI_Tensor_RdbSave(RedisModuleIO *io, void *value) {
    RAI_RDBSaveTensor(io, value);
}

static void* RAI_Tensor_RdbLoad(struct RedisModuleIO *io, int encver) {
  // if (encver != RAI_ENC_VER) {
  //   /* We should actually log an error here, or try to implement
  //      the ability to load older versions of our data structure. */
  //   return NULL;
  // }

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

static void RAI_Tensor_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_AofRewriteTensor(aof, key, value);
}

static void RAI_Tensor_DTFree(void *value) {
    RAI_TensorFree(value);
}

int TensorType_Register(RedisModuleCtx *ctx) {
    RedisModuleTypeMethods tmTensor = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RAI_Tensor_RdbLoad,
      .rdb_save = RAI_Tensor_RdbSave,
      .aof_rewrite = RAI_Tensor_AofRewrite,
      .mem_usage = NULL,
      .free = RAI_Tensor_DTFree,
      .digest = NULL,
  };
  RedisAI_TensorType = RedisModule_CreateDataType(ctx, "AI_TENSOR", RAI_ENC_VER_MM, &tmTensor);
  return RedisAI_TensorType != NULL;
}
