#include "tensor.h"
#include "tensor_struct.h"
#include <stddef.h>
#include <strings.h>
#include <string.h>

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

static void* Tensor_RdbLoad(struct RedisModuleIO *io, int encver){
  //todo
  return NULL;
}

static void Tensor_RdbSave(RedisModuleIO *rdb, void *value){
  //todo
}

static void Tensor_DTFree(void *value){
  RAI_TensorFree(value);
}

int RAI_TensorInit(RedisModuleCtx* ctx){
  RedisModuleTypeMethods tmTensor = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = Tensor_RdbLoad,
      .rdb_save = Tensor_RdbSave,
      .aof_rewrite = NULL,
      .mem_usage = NULL,
      .free = Tensor_DTFree,
      .digest = NULL,
  };
  RedisAI_TensorType = RedisModule_CreateDataType(ctx, "DL_TENSOR", 0, &tmTensor);
  return RedisAI_TensorType != NULL;
}

RAI_Tensor* RAI_TensorCreate(const char* dataTypeStr, long long* dims, int ndims) {
  DLDataType dtype = Tensor_GetDataType(dataTypeStr);

  if (Tensor_DataTypeSize(dtype) == 0){
    return NULL;
  }

  RAI_Tensor* ret = RedisModule_Alloc(sizeof(*ret));

  long long* shape = RedisModule_Alloc(ndims * sizeof(*shape));
  long long* strides = RedisModule_Alloc(ndims * sizeof(*strides));
  size_t len = 1;
  for (long long i = 0 ; i < ndims ; ++i){
    shape[i] = dims[i];
    strides[i] = 1;
    len *= dims[i];
  }

  DLContext ctx = (DLContext){
      .device_type = kDLCPU,
      .device_id = 0
  };

  void* data = RedisModule_Alloc(len * Tensor_DataTypeSize(dtype));

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

  RAI_Tensor* ret = RedisModule_Alloc(sizeof(*ret));

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

DLDataType RAI_TensorDataType(RAI_Tensor* t){
  return t->tensor.dl_tensor.dtype;
}

size_t RAI_TensorLength(RAI_Tensor* t) {
  long long* shape = t->tensor.dl_tensor.shape;
  size_t len = 1;
  for (size_t i = 0 ; i < t->tensor.dl_tensor.ndim; ++i){
    len *= shape[i];
  }
  return len;
}

size_t RAI_TensorGetDataSize(const char* dataTypeStr){
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
  return Tensor_DataTypeSize(RAI_TensorDataType(t)) * RAI_TensorLength(t);
}

char* RAI_TensorData(RAI_Tensor* t){
  return t->tensor.dl_tensor.data;
}
