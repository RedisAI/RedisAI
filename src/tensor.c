#include "tensor.h"
#include "tensor_struct.h"
#include <stddef.h>
#include <strings.h>
#include <string.h>

RedisModuleType *RedisDL_TensorType = NULL;

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
  RDL_TensorFree(value);
}

int RDL_TensorInit(RedisModuleCtx* ctx){
  RedisModuleTypeMethods tmTensor = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = Tensor_RdbLoad,
      .rdb_save = Tensor_RdbSave,
      .aof_rewrite = NULL,
      .mem_usage = NULL,
      .free = Tensor_DTFree,
      .digest = NULL,
  };
  RedisDL_TensorType = RedisModule_CreateDataType(ctx, "DL_TENSOR", 0, &tmTensor);
  return RedisDL_TensorType != NULL;
}

RDL_Tensor* RDL_TensorCreate(const char* dataTypeStr, long long* dims, int ndims) {
  DLDataType dtype = Tensor_GetDataType(dataTypeStr);

  if (Tensor_DataTypeSize(dtype) == 0){
    return NULL;
  }

  RDL_Tensor* ret = RedisModule_Alloc(sizeof(*ret));

  long long* shape = RedisModule_Alloc(ndims * sizeof(*shape));
  size_t len = 1;
  for (long long i = 0 ; i < ndims ; ++i){
    shape[i] = dims[i];
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

  ret->tensor = (DLTensor){
      .ctx = ctx,
      .data = data,
      .ndim = ndims,
      .dtype = dtype,
      .shape = shape,
      .strides = NULL,
      .byte_offset = 0
  };

  ret->refCount = 1;
  return ret;
}

DLDataType RDL_TensorDataType(RDL_Tensor* t){
  return t->tensor.dtype;
}

size_t RDL_TensorLength(RDL_Tensor* t) {
  long long* shape = t->tensor.shape;
  size_t len = 1;
  for (size_t i = 0 ; i < t->tensor.ndim; ++i){
    len *= shape[i];
  }
  return len;
}

size_t RDL_TensorGetDataSize(const char* dataTypeStr){
  DLDataType dtype = Tensor_GetDataType(dataTypeStr);
  return Tensor_DataTypeSize(dtype);
}

void RDL_TensorFree(RDL_Tensor* t){
  if (--t->refCount <= 0){
    RedisModule_Free(t->tensor.shape);
    RedisModule_Free(t->tensor.data);
    RedisModule_Free(t);
  }
}

int RDL_TensorSetData(RDL_Tensor* t, const char* data, size_t len){
  memcpy(t->tensor.data, data, len);
  return 1;
}

int RDL_TensorSetValueFromLongLong(RDL_Tensor* t, long long i, long long val){
  DLDataType dtype = t->tensor.dtype;
  void* data = t->tensor.data;

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

int RDL_TensorSetValueFromDouble(RDL_Tensor* t, long long i, double val){
  DLDataType dtype = t->tensor.dtype;
  void* data = t->tensor.data;

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

int RDL_TensorGetValueAsDouble(RDL_Tensor* t, long long i, double* val) {
  DLDataType dtype = t->tensor.dtype;
  void* data = t->tensor.data;

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

int RDL_TensorGetValueAsLongLong(RDL_Tensor* t, long long i, long long* val) {
  DLDataType dtype = t->tensor.dtype;
  void* data = t->tensor.data;

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

RDL_Tensor* RDL_TensorGetShallowCopy(RDL_Tensor* t){
  ++t->refCount;
  return t;
}

int RDL_TensorNumDims(RDL_Tensor* t){
  return t->tensor.ndim;
}

long long RDL_TensorDim(RDL_Tensor* t, int i){
  return t->tensor.shape[i];
}

size_t RDL_TensorByteSize(RDL_Tensor* t){
  return Tensor_DataTypeSize(RDL_TensorDataType(t)) * RDL_TensorLength(t);
}

char* RDL_TensorData(RDL_Tensor* t){
  return t->tensor.data;
}
