#include "tensor.h"
#include "tensor_struct.h"
#include <stddef.h>
#include <strings.h>
#include <string.h>
#include "rmutil/alloc.h"
#include <assert.h>

RedisModuleType *RedisAI_TensorType = NULL;

DLDataType RAI_TensorDataTypeFromString(const char* typestr){
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

void Tensor_DataTypeStr(DLDataType dtype, char **dtypestr) {
  *dtypestr = RedisModule_Calloc(8, sizeof(char));
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 32) {
      strcpy(*dtypestr, "FLOAT");
    }
    else if (dtype.bits == 64) {
      strcpy(*dtypestr, "DOUBLE");
    }
    else {
      RedisModule_Free(*dtypestr);
      *dtypestr = NULL;
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
    else {
      RedisModule_Free(*dtypestr);
      *dtypestr = NULL;
    }
  }
  else if (dtype.code == kDLUInt) {
    if (dtype.bits == 8) {
      strcpy(*dtypestr, "UINT8");
    }
    else if (dtype.bits == 16) {
      strcpy(*dtypestr, "UINT16");
    }
    else {
      RedisModule_Free(*dtypestr);
      *dtypestr = NULL;
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

  char *data = RAI_TensorData(tensor);
  long long size = RAI_TensorByteSize(tensor);

  long long ndims = RAI_TensorNumDims(tensor);

  RedisModuleString* dims[ndims];

  for (long long i=0; i<ndims; i++) {
    dims[i] = RedisModule_CreateStringFromLongLong(RedisModule_GetContextFromIO(aof), RAI_TensorDim(tensor, i));
  }

  RedisModule_EmitAOF(aof, "AI.TENSORSET", "scvcb", key, dtypestr, dims, ndims, "BLOB", data, size);
 
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

RAI_Tensor* RAI_TensorCreateWithDLDataType(DLDataType dtype, long long* dims, int ndims, int hasdata) {
  const size_t dtypeSize = Tensor_DataTypeSize(dtype);
  if ( dtypeSize == 0){
    return NULL;
  }

  RAI_Tensor* ret = RedisModule_Alloc(sizeof(*ret));
  int64_t* shape = RedisModule_Alloc(ndims*sizeof(*shape));
  int64_t* strides = RedisModule_Alloc(ndims*sizeof(*strides));

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
  void* data = NULL;
  if (hasdata) {
    data = RedisModule_Alloc(len * dtypeSize);
  }
  else {
    data = RedisModule_Calloc(len, dtypeSize);
  }

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

RAI_Tensor* RAI_TensorCreate(const char* dataType, long long* dims, int ndims, int hasdata) {
  DLDataType dtype = RAI_TensorDataTypeFromString(dataType);
  return RAI_TensorCreateWithDLDataType(dtype, dims, ndims, hasdata);
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

RAI_Tensor* RAI_TensorCreateByConcatenatingTensors(RAI_Tensor** ts, long long n) {

  if (n == 0) {
    return NULL;
  }

  long long total_batch_size = 0;
  long long batch_sizes[n];
  long long batch_offsets[n];

  long long ndims = RAI_TensorNumDims(ts[0]);
  long long dims[ndims];

  // TODO check that all tensors have compatible dims

  for (long long i=0; i<n; i++) {
    batch_sizes[i] = RAI_TensorDim(ts[i], 0);
    total_batch_size += batch_sizes[i];
  }

  batch_offsets[0] = 0;
  for (long long i=1; i<n; i++) {
    batch_offsets[i] = batch_sizes[i-1];
  }

  long long sample_size = 0;

  for (long long i=1; i<ndims; i++) {
    dims[i] = RAI_TensorDim(ts[0], i);
    sample_size *= dims[i];
  }
  dims[0] = total_batch_size;

  long long dtype_size = RAI_TensorDataSize(ts[0]);

  DLDataType dtype = RAI_TensorDataType(ts[0]);

  RAI_Tensor* ret = RAI_TensorCreateWithDLDataType(dtype, dims, ndims, 1);

  for (long long i=0; i<n; i++) {
    memcpy(RAI_TensorData(ret) + batch_offsets[i] * sample_size * dtype_size, RAI_TensorData(ts[i]), RAI_TensorByteSize(ts[i]));
  }

  return ret;
}

RAI_Tensor* RAI_TensorCreateBySlicingTensor(RAI_Tensor* t, long long offset, long long len) {

  long long ndims = RAI_TensorNumDims(t);
  long long dims[ndims];

  long long dtype_size = RAI_TensorDataSize(t);
  long long sample_size = 1;

  for (long long i=1; i<ndims; i++) {
    dims[i] = RAI_TensorDim(t, i);
    sample_size *= dims[i];
  }

  dims[0] = len;

  DLDataType dtype = RAI_TensorDataType(t);

  RAI_Tensor* ret = RAI_TensorCreateWithDLDataType(dtype, dims, ndims, 1);

  memcpy(RAI_TensorData(ret), RAI_TensorData(t) + offset * sample_size * dtype_size, len * sample_size * dtype_size);

  return ret;
}

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

size_t RAI_TensorDataSize(RAI_Tensor* t) {
  return Tensor_DataTypeSize(RAI_TensorDataType(t));
}

size_t RAI_TensorDataSizeFromString(const char* dataTypeStr) {
  DLDataType dtype = RAI_TensorDataTypeFromString(dataTypeStr);
  return Tensor_DataTypeSize(dtype);
}

size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype) {
  return Tensor_DataTypeSize(dtype);
}

void RAI_TensorFree(RAI_Tensor* t){
  if (--t->refCount <= 0){
    if (t->tensor.deleter) {
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
