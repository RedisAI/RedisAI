#include "tensor.h"
#include "tensor_struct.h"
#include <stddef.h>
#include <strings.h>
#include <string.h>
#include "util/alloc.h"
#include <assert.h>

RedisModuleType *RedisAI_TensorType = NULL;

static DLDataType Tensor_GetDataType(const char* typestr){
  if (strcasecmp(typestr, "FLOAT") == 0){
    return (DLDataType){ .code = kDLFloat, .bits = 32, .lanes = 1};
  }
  if (strcasecmp(typestr, "DOUBLE") == 0) {
    return (DLDataType){ .code = kDLFloat, .bits = 64, .lanes = 1};
  }
  if (strncasecmp(typestr, "INT", 3) == 0) {
    const char *bitstr = typestr + 3;
    if (strcmp(bitstr, "8") == 0){
      return (DLDataType){.code = kDLInt, .bits = 8, .lanes = 1};
    }
    if (strcmp(bitstr, "16") == 0){
      return (DLDataType){.code = kDLInt, .bits = 16, .lanes = 1};
    }
    if (strcmp(bitstr, "32") == 0){
      return (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1};
    }
    if (strcmp(bitstr, "64") == 0){
      return (DLDataType){.code = kDLInt, .bits = 64, .lanes = 1};
    }
  }
  if (strncasecmp(typestr, "UINT", 4) == 0) {
    const char *bitstr = typestr + 4;
    if (strcmp(bitstr, "8") == 0){
      return (DLDataType){.code = kDLUInt, .bits = 8, .lanes = 1};
    }
    if (strcmp(bitstr, "16") == 0){
      return (DLDataType){.code = kDLUInt, .bits = 16, .lanes = 1};
    }
  }
  return (DLDataType){.bits = 0};
}

static size_t Tensor_DataTypeSize(DLDataType dtype) {
  return dtype.bits / 8;
}

static void Tensor_DataTypeStr(DLDataType dtype, char **dtypestr) {
  *dtypestr = RedisModule_Calloc(8, sizeof(char));
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 32) {
      strcpy(*dtypestr, "FLOAT32");
    }
    else if (dtype.bits == 64) {
      strcpy(*dtypestr, "FLOAT64");
    }
  }
  else if (dtype.code == kDLInt) {
    if (dtype.bits == 8) {
      strcpy(*dtypestr, "INT8");
    }
    else if (dtype.bits == 16) {
      strcpy(*dtypestr, "INT16");
    }
    else if (dtype.bits == 32) {
      strcpy(*dtypestr, "INT32");
    }
    else if (dtype.bits == 64) {
      strcpy(*dtypestr, "INT64");
    }
  }
  else if (dtype.code == kDLUInt) {
    if (dtype.bits == 8) {
      strcpy(*dtypestr, "UINT8");
    }
    else if (dtype.bits == 16) {
      strcpy(*dtypestr, "UINT16");
    }
  }
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

  RAI_Tensor *ret = RedisModule_Calloc(1, sizeof(*ret));

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

static void RAI_Tensor_RdbSave(RedisModuleIO *io, void *value) {
  RAI_Tensor *tensor = (RAI_Tensor*)value;

  size_t ndim = tensor->tensor.dl_tensor.ndim;

  RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.ctx.device_type);
  RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.ctx.device_id);
  RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.bits);
  RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.code);
  RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.lanes);
  RedisModule_SaveUnsigned(io, ndim);
  for (size_t i=0; i<ndim; i++) {
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.shape[i]);
  }
  for (size_t i=0; i<ndim; i++) {
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.strides[i]);
  }
  RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.byte_offset);
  size_t size = RAI_TensorByteSize(tensor);

  RedisModule_SaveStringBuffer(io, tensor->tensor.dl_tensor.data, size);
}

#define RAI_SPLICE_SHAPE_1(x) x[0]
#define RAI_SPLICE_SHAPE_2(x) x[0], x[1]
#define RAI_SPLICE_SHAPE_3(x) x[0], x[1], x[2]
#define RAI_SPLICE_SHAPE_4(x) x[0], x[1], x[2], x[3]
#define RAI_SPLICE_SHAPE_5(x) x[0], x[1], x[2], x[3], x[4]
#define RAI_SPLICE_SHAPE_6(x) x[0], x[1], x[2], x[3], x[4], x[5]
#define RAI_SPLICE_SHAPE_7(x) x[0], x[1], x[2], x[3], x[4], x[5], x[6]
#define RAI_SPLICE_SHAPE_8(x) x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]

// AI.TENSORSET tensor_key data_type shape1 shape2 ... BLOB data

static void RAI_Tensor_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  RAI_Tensor *tensor = (RAI_Tensor*)value;

  char *dtypestr = NULL;

  Tensor_DataTypeStr(RAI_TensorDataType(tensor), &dtypestr);

  int64_t* shape = tensor->tensor.dl_tensor.shape;
  char* data = RAI_TensorData(tensor);
  size_t size = RAI_TensorByteSize(tensor);

  // We switch over the dimensions of the tensor up to 7
  // The reason is that we don't have a way to pass a vector of long long to RedisModule_EmitAOF,
  // there's no format for it. Vector of strings is supported (format 'v').
  // This might change in the future, but it needs to change in redis/src/module.c

  switch (RAI_TensorNumDims(tensor)) {
    case 1:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "sllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_1(shape), "BLOB", data, size);
      break;
    case 2:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "slllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_2(shape), "BLOB", data, size);
      break;
    case 3:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "sllllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_3(shape), "BLOB", data, size);
      break;
    case 4:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "slllllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_4(shape), "BLOB", data, size);
      break;
    case 5:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "sllllllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_5(shape), "BLOB", data, size);
      break;
    case 6:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "slllllllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_6(shape), "BLOB", data, size);
      break;
    case 7:
      RedisModule_EmitAOF(aof, "AI.TENSORSET", "sllllllllcb",
                          key, dtypestr, RAI_SPLICE_SHAPE_7(shape), "BLOB", data, size);
      break;
    default:
      printf("ERR: AOF serialization supports tensors of dimension up to 7\n");
  }

  RedisModule_Free(dtypestr);
}

static void RAI_Tensor_DTFree(void *value) {
  RAI_TensorFree(value);
}

int RAI_TensorInit(RedisModuleCtx* ctx){
  RedisModuleTypeMethods tmTensor = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RAI_Tensor_RdbLoad,
      .rdb_save = RAI_Tensor_RdbSave,
      .aof_rewrite = RAI_Tensor_AofRewrite,
      .mem_usage = NULL,
      .free = RAI_Tensor_DTFree,
      .digest = NULL,
  };
  RedisAI_TensorType = RedisModule_CreateDataType(ctx, "AI_TENSOR", 0, &tmTensor);
  return RedisAI_TensorType != NULL;
}

RAI_Tensor* RAI_TensorCreate(const char* dataTypeStr, long long* dims, int ndims) {
  DLDataType dtype = Tensor_GetDataType(dataTypeStr);

  if (Tensor_DataTypeSize(dtype) == 0){
    return NULL;
  }

  RAI_Tensor* ret = RedisModule_Calloc(1, sizeof(*ret));

  int64_t* shape = RedisModule_Calloc(ndims, sizeof(*shape));
  int64_t* strides = RedisModule_Calloc(ndims, sizeof(*strides));
  size_t len = 1;
  for (int64_t i = 0 ; i < ndims ; ++i){
    shape[i] = dims[i];
    strides[i] = 1;
    len *= dims[i];
  }
  for (int64_t i = ndims-2 ; i >= 0 ; --i) {
    strides[i] *= strides[i+1] * shape[i+1];
  }

  DLContext ctx = (DLContext){
      .device_type = kDLCPU,
      .device_id = 0
  };

  void* data = RedisModule_Calloc(len, Tensor_DataTypeSize(dtype));

  if (data == NULL) {
    RedisModule_Free(ret);
    return NULL;
  }

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

#if 0
void RAI_TensorMoveFrom(RAI_Tensor* dst, RAI_Tensor* src) {
  if (--dst->refCount <= 0){
    RedisModule_Free(t->tensor.shape);
    if (t->tensor.strides) {
      RedisModule_Free(t->tensor.strides);
    }
    RedisModule_Free(t->tensor.data);
    RedisModule_Free(t);
  }
  dst->tensor.ctx = src->tensor.ctx;
  dst->tensor.data = src->tensor.data;

  dst->refCount = 1;
}
#endif

// Beware: this will take ownership of dltensor
RAI_Tensor* RAI_TensorCreateFromDLTensor(DLManagedTensor* dl_tensor) {

  RAI_Tensor* ret = RedisModule_Calloc(1, sizeof(*ret));

  ret->tensor = (DLManagedTensor){
    .dl_tensor = (DLTensor){
      .ctx = dl_tensor->dl_tensor.ctx,
      .data = dl_tensor->dl_tensor.data,
      .ndim = dl_tensor->dl_tensor.ndim,
      .dtype = dl_tensor->dl_tensor.dtype,
      .shape = dl_tensor->dl_tensor.shape,
      .strides = dl_tensor->dl_tensor.strides,
      .byte_offset = dl_tensor->dl_tensor.byte_offset
    },
    .manager_ctx = dl_tensor->manager_ctx,
    .deleter  = dl_tensor->deleter
  };

  ret->refCount = 1;
  return ret;
}

DLDataType RAI_TensorDataType(RAI_Tensor* t) {
  return t->tensor.dl_tensor.dtype;
}

size_t RAI_TensorLength(RAI_Tensor* t) {
  int64_t* shape = t->tensor.dl_tensor.shape;
  size_t len = 1;
  for (size_t i = 0 ; i < t->tensor.dl_tensor.ndim; ++i){
    len *= shape[i];
  }
  return len;
}

size_t RAI_TensorGetDataSize(const char* dataTypeStr) {
  DLDataType dtype = Tensor_GetDataType(dataTypeStr);
  return Tensor_DataTypeSize(dtype);
}

void RAI_TensorFree(RAI_Tensor* t){
  if (--t->refCount <= 0){
    if (t->tensor.manager_ctx && t->tensor.deleter) {
      t->tensor.deleter(&t->tensor);
    }
    else {
      RedisModule_Free(t->tensor.dl_tensor.shape);
      if (t->tensor.dl_tensor.strides) {
        RedisModule_Free(t->tensor.dl_tensor.strides);
      }
      RedisModule_Free(t->tensor.dl_tensor.data);
    }
    RedisModule_Free(t);
  }
}

int RAI_TensorSetData(RAI_Tensor* t, const char* data, size_t len){
  memcpy(t->tensor.dl_tensor.data, data, len);
  return 1;
}

int RAI_TensorSetValueFromLongLong(RAI_Tensor* t, long long i, long long val){
  DLDataType dtype = t->tensor.dl_tensor.dtype;
  void* data = t->tensor.dl_tensor.data;

  if (dtype.code == kDLInt) {
    switch (dtype.bits) {
      case 8:
        ((int8_t *)data)[i] = val; break;
        break;
      case 16:
        ((int16_t *)data)[i] = val; break;
        break;
      case 32:
        ((int32_t *)data)[i] = val; break;
        break;
      case 64:
        ((int64_t *)data)[i] = val; break;
        break;
      default:
        return 0;
    }
  }
  else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
      case 8:
        ((uint8_t *)data)[i] = val; break;
        break;
      case 16:
        ((uint16_t *)data)[i] = val; break;
        break;
      case 32:
        ((uint32_t *)data)[i] = val; break;
        break;
      case 64:
        ((uint64_t *)data)[i] = val; break;
        break;
      default:
        return 0;
    }
  }
  else {
    return 0;
  }
  return 1;
}

int RAI_TensorSetValueFromDouble(RAI_Tensor* t, long long i, double val){
  DLDataType dtype = t->tensor.dl_tensor.dtype;
  void* data = t->tensor.dl_tensor.data;

  if (dtype.code == kDLFloat) {
    switch (dtype.bits) {
      case 32:
        ((float *)data)[i] = val; break;
      case 64:
        ((double *)data)[i] = val; break;
      default:
        return 0;
    }
  }
  else {
    return 0;
  }
  return 1;
}

int RAI_TensorGetValueAsDouble(RAI_Tensor* t, long long i, double* val) {
  DLDataType dtype = t->tensor.dl_tensor.dtype;
  void* data = t->tensor.dl_tensor.data;

  // TODO: check i is in bound
  if (dtype.code == kDLFloat) {
    switch (dtype.bits) {
      case 32:
        *val = ((float *)data)[i]; break;
      case 64:
        *val = ((double *)data)[i]; break;
      default:
        return 0;
    }
  }
  else {
    return 0;
  }
  return 1;
}

int RAI_TensorGetValueAsLongLong(RAI_Tensor* t, long long i, long long* val) {
  DLDataType dtype = t->tensor.dl_tensor.dtype;
  void* data = t->tensor.dl_tensor.data;

  // TODO: check i is in bound

  if (dtype.code == kDLInt) {
    switch (dtype.bits) {
      case 8:
        *val = ((int8_t *)data)[i]; break;
      case 16:
        *val = ((int16_t *)data)[i]; break;
      case 32:
        *val = ((int32_t *)data)[i]; break;
      case 64:
        *val = ((int64_t *)data)[i]; break;
      default:
        return 0;
    }
  }
  else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
      case 8:
        *val = ((uint8_t *)data)[i]; break;
      case 16:
        *val = ((uint16_t *)data)[i]; break;
      case 32:
        *val = ((uint32_t *)data)[i]; break;
      case 64:
        *val = ((uint64_t *)data)[i]; break;
      default:
        return 0;
    }
  }
  else {
    return 0;
  }
  return 1;
}

RAI_Tensor* RAI_TensorGetShallowCopy(RAI_Tensor* t){
  ++t->refCount;
  return t;
}

int RAI_TensorNumDims(RAI_Tensor* t){
  return t->tensor.dl_tensor.ndim;
}

long long RAI_TensorDim(RAI_Tensor* t, int i){
  return t->tensor.dl_tensor.shape[i];
}

size_t RAI_TensorByteSize(RAI_Tensor* t){
  // TODO: as per dlpack it should be
  //   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
  return Tensor_DataTypeSize(RAI_TensorDataType(t)) * RAI_TensorLength(t);
}

char* RAI_TensorData(RAI_Tensor* t){
  return t->tensor.dl_tensor.data;
}
