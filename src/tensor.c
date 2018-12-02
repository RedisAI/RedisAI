#include "tensor.h"
#include <stddef.h>
#include <strings.h>
#include <string.h>

RedisModuleType *RedisDL_TensorType = NULL;

typedef struct RDL_Tensor {
  TF_Tensor* tensor;
  size_t refCount;
}RDL_Tensor;

static TF_DataType Tensor_GetDataType(const char* typestr){
  if (strcasecmp(typestr, "FLOAT") == 0){
    return TF_FLOAT;
  }
  if (strcasecmp(typestr, "DOUBLE") == 0) {
    return TF_DOUBLE;
  }
  if (strncasecmp(typestr, "INT", 3) == 0) {
    const char *bitstr = typestr + 3;
    if (strcmp(bitstr, "8") == 0){
      return TF_INT8;
    }
    if (strcmp(bitstr, "16") == 0){
      return TF_INT16;
    }
    if (strcmp(bitstr, "32") == 0){
      return TF_INT32;
    }
    if (strcmp(bitstr, "64") == 0){
      return TF_INT64;
    }
    return 0;
  }
  if (strncasecmp(typestr, "UINT", 4) == 0) {
    const char *bitstr = typestr + 4;
    if (strcmp(bitstr, "8") == 0){
      return TF_UINT8;
    }
    if (strcmp(bitstr, "16") == 0){
      return TF_UINT16;
    }
    return 0;
  }
  return 0;
}

static size_t Tensor_DataSize(TF_DataType type) {
  switch (type) {
    case TF_BOOL:
      return sizeof(int8_t);
    case TF_INT8:
      return sizeof(int8_t);
    case TF_UINT8:
      return sizeof(uint8_t);
    case TF_INT16:
      return sizeof(int16_t);
    case TF_UINT16:
      return sizeof(uint16_t);
    case TF_INT32:
      return sizeof(int32_t);
    case TF_INT64:
      return sizeof(int64_t);
    case TF_FLOAT:
      return sizeof(float);
    case TF_DOUBLE:
      return sizeof(double);
    case TF_COMPLEX64:
      return 2*sizeof(float);
    case TF_COMPLEX128:
      return 2 * sizeof(double);
    default:
      return 0;
  }
}

static void* Tensor_RdbLoad(struct RedisModuleIO *io, int encver){
  //todo
  return NULL;
}

static void Tensor_RdbSave(RedisModuleIO *rdb, void *value){
  //todo
}

static void Tensor_DTFree(void *value){
  Tensor_Free(value);
}

int Tensor_Init(RedisModuleCtx* ctx){
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

RDL_Tensor* Tensor_Create(const char* dataTypeStr, long long* dims,int ndims){
  TF_DataType dtype = Tensor_GetDataType(dataTypeStr);
  if(!dtype){
    return NULL;
  }

  RDL_Tensor* ret = RedisModule_Alloc(sizeof(*ret));
  size_t len = 1;
  for(long long i = 0 ; i < ndims ; ++i){
    len *= dims[i];
  }
  ret->tensor = TF_AllocateTensor(dtype, (const int64_t*)dims, ndims, len * Tensor_DataSize(dtype));
  if(!ret->tensor){
    RedisModule_Free(ret);
    return NULL;
  }
  ret->refCount = 1;
  return ret;
}

RDL_Tensor* Tensor_CreateFromTensor(TF_Tensor *tensor){
  RDL_Tensor* ret = RedisModule_Alloc(sizeof(*ret));
  ret->tensor = tensor;
  ret->refCount = 1;
  return ret;
}

TF_DataType Tensor_DataType(RDL_Tensor* t){
  return TF_TensorType(t->tensor);
}

size_t Tensor_GetDataSize(const char* dataTypeStr){
  TF_DataType dtype = Tensor_GetDataType(dataTypeStr);
  if(!dtype){
    return 0;
  }
  return Tensor_DataSize(dtype);
}

void Tensor_Free(RDL_Tensor* t){
  if(--t->refCount <= 0){
    TF_DeleteTensor(t->tensor);
    RedisModule_Free(t);
  }
}

int Tensor_SetData(RDL_Tensor* t, const char* data, size_t len){
  memcpy(TF_TensorData(t->tensor), data, len);
  return 1;
}

int Tensor_SetValueFromLongLong(RDL_Tensor* t, long long i, long long val){
  switch (TF_TensorType(t->tensor)) {
    case TF_BOOL:
      ((int8_t*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_INT8:
      ((int8_t*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_UINT8:
      ((uint8_t*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_INT16:
      ((int16_t*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_UINT16:
      ((uint16_t*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_INT32:
      ((int32_t*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_INT64:
      ((int64_t*)TF_TensorData(t->tensor))[i] = val; break;
    default:
      return 0;
  }
  return 1;
}

int Tensor_SetValueFromDouble(RDL_Tensor* t, long long i, double val){
  switch (TF_TensorType(t->tensor)) {
    case TF_FLOAT:
      ((float*)TF_TensorData(t->tensor))[i] = val; break;
    case TF_DOUBLE:
      ((double*)TF_TensorData(t->tensor))[i] = val; break;
    default:
      return 0;
  }
  return 1;
}

int Tensor_GetValueAsDouble(RDL_Tensor* t, long long i, double* val) {
  switch (TF_TensorType(t->tensor)) {
    case TF_FLOAT:
      *val = ((float*)TF_TensorData(t->tensor))[i]; break;
    case TF_DOUBLE:
      *val = ((double*)TF_TensorData(t->tensor))[i]; break;
    default:
      return 0;
  }
  return 1;
}

int Tensor_GetValueAsLongLong(RDL_Tensor* t, long long i, long long* val) {
  switch (TF_TensorType(t->tensor)) {
    case TF_BOOL:
      *val = ((int8_t*)TF_TensorData(t->tensor))[i]; break;
    case TF_INT8:
      *val = ((int8_t*)TF_TensorData(t->tensor))[i]; break;
    case TF_UINT8:
      *val = ((uint8_t*)TF_TensorData(t->tensor))[i]; break;
    case TF_INT16:
      *val = ((int16_t*)TF_TensorData(t->tensor))[i]; break;
    case TF_UINT16:
      *val = ((uint16_t*)TF_TensorData(t->tensor))[i]; break;
    case TF_INT32:
      *val = ((int32_t*)TF_TensorData(t->tensor))[i]; break;
    case TF_INT64:
      *val = ((int64_t*)TF_TensorData(t->tensor))[i]; break;
    default:
      return 0;
  }
  return 1;
}

RDL_Tensor* Tensor_GetShallowCopy(RDL_Tensor* t){
  ++t->refCount;
  return t;
}

int Tensor_NumDims(RDL_Tensor* t){
  return TF_NumDims(t->tensor);
}

long long Tensor_Dim(RDL_Tensor* t, int dim){
  return TF_Dim(t->tensor, dim);
}

size_t Tensor_ByteSize(RDL_Tensor* t){
  return TF_TensorByteSize(t->tensor);
}

char* Tensor_Data(RDL_Tensor* t){
  return TF_TensorData(t->tensor);
}

TF_Tensor* Tensor_GetTensor(RDL_Tensor* t){
  return t->tensor;
}
