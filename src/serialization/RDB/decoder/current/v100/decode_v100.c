#include "decode_v100.h"
#include "assert.h"

void* RAI_RDBLoadTensor_v100(RedisModuleIO *io) {
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