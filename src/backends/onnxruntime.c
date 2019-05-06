#include "backends/onnxruntime.h"
#include "tensor.h"
#include "util/alloc.h"
#include "util/arr_rm_alloc.h"

#if 0
TF_DataType RAI_GetTFDataTypeFromDL(DLDataType dtype) {

  if (dtype.code == kDLFloat) {
    switch (dtype.bits) {
      case 32:
        return TF_FLOAT; break;
      case 64:
        return TF_DOUBLE; break;
      default:
        return 0;
    }
  }
  else if (dtype.code == kDLInt) {
    switch (dtype.bits) {
      case 8:
        return TF_INT8; break;
      case 16:
        return TF_INT16; break;
      case 32:
        return TF_INT32; break;
      case 64:
        return TF_INT64; break;
      default:
        return 0;
    }
  }
  else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
      case 8:
        return TF_UINT8; break;
      case 16:
        return TF_UINT16; break;
      default:
        return 0;
    }
  }
  return 0;
}

DLDataType RAI_GetDLDataTypeFromTF(TF_DataType dtype) {
  switch (dtype) {
    case TF_FLOAT:
      return (DLDataType){ .code = kDLFloat, .bits = 32, .lanes = 1 };
    case TF_DOUBLE:
      return (DLDataType){ .code = kDLFloat, .bits = 64, .lanes = 1 };
    case TF_INT8:
      return (DLDataType){ .code = kDLInt, .bits = 8, .lanes = 1 };
    case TF_INT16:
      return (DLDataType){ .code = kDLInt, .bits = 16, .lanes = 1 };
    case TF_INT32:
      return (DLDataType){ .code = kDLInt, .bits = 32, .lanes = 1 };
    case TF_INT64:
      return (DLDataType){ .code = kDLInt, .bits = 64, .lanes = 1 };
    case TF_UINT8:
      return (DLDataType){ .code = kDLUInt, .bits = 8, .lanes = 1 };
    case TF_UINT16:
      return (DLDataType){ .code = kDLUInt, .bits = 16, .lanes = 1 };
    default:
      return (DLDataType){ .bits = 0 };
  }
  return (DLDataType){ .bits = 0 };
}

RAI_Tensor* RAI_TensorCreateFromTFTensor(TF_Tensor *tensor) {
  RAI_Tensor* ret = RedisModule_Calloc(1, sizeof(*ret));

  DLContext ctx = (DLContext){
      .device_type = kDLCPU,
      .device_id = 0
  };

  size_t ndims = TF_NumDims(tensor);

  int64_t* shape = RedisModule_Calloc(ndims, sizeof(*shape));
  int64_t* strides = RedisModule_Calloc(ndims, sizeof(*strides));
  for (int64_t i = 0 ; i < ndims ; ++i) {
    shape[i] = TF_Dim(tensor, i);
    strides[i] = 1;
  }
  for (int64_t i = ndims-2 ; i >= 0 ; --i) {
    strides[i] *= strides[i+1] * shape[i+1];
  }

  // FIXME: In TF, RunSession allocates memory for output tensors
  // This means that we either memcpy the tensor data and let
  // Redis be responsible for the memory, or we reuse the TF
  // allocated memory, which might not be optimal down the road
  // Note: on YOLO this has no impact on perf
#ifdef RAI_COPY_RUN_OUTPUT
  size_t len = TF_TensorByteSize(tensor);
  char* data = RedisModule_Calloc(len, sizeof(*data));
  memcpy(data, TF_TensorData(tensor), len);
#endif

  // TODO: use manager_ctx to ensure TF tensor doesn't get deallocated
  // This applies to outputs

  ret->tensor = (DLManagedTensor){
    .dl_tensor = (DLTensor){
      .ctx = ctx,
#ifdef RAI_COPY_RUN_OUTPUT
      .data = data,
#else
      .data = TF_TensorData(tensor),
#endif
      .ndim = ndims,
      .dtype = RAI_GetDLDataTypeFromTF(TF_TensorType(tensor)),
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

void RAI_TFDeallocator(void* data, size_t len, void* arg) {
  // printf("DEALLOCATOR CALLED\n");
  // do nothing, memory is managed by Redis
}

TF_Tensor* RAI_TFTensorFromTensor(RAI_Tensor* t){
#ifdef RAI_COPY_RUN_INPUT
  TF_Tensor* out = TF_AllocateTensor(
      RAI_GetTFDataTypeFromDL(t->tensor.dl_tensor.dtype),
      t->tensor.dl_tensor.shape,
      t->tensor.dl_tensor.ndim,
      RAI_TensorByteSize(t));
  memcpy(TF_TensorData(out), t->tensor.dl_tensor.data, TF_TensorByteSize(out));
  return out;
#else
  return TF_NewTensor(
      RAI_GetTFDataTypeFromDL(t->tensor.dl_tensor.dtype),
      t->tensor.dl_tensor.shape,
      t->tensor.dl_tensor.ndim,
      t->tensor.dl_tensor.data,
      RAI_TensorByteSize(t),
      &RAI_TFDeallocator,
      NULL);
#endif /* RAI_COPY_RUN_INPUT */
}

#endif

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, RAI_Device device,
                              const char *modeldef, size_t modellen,
                              RAI_Error *error) {

  // TODO: take from
  // https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

  OrtStatus* onnx_status;

  OrtEnv* env;
  onnx_status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

  if (onnx_status != NULL) {
    const char* msg = OrtGetErrorMessage(onnx_status);
    OrtReleaseStatus(onnx_status);
    // TODO fill err
    return NULL;
  }

  // TODO: probably these options could be configured at the AI.CONFIG level
  OrtSessionOptions* session_options = OrtCreateSessionOptions();
  // OrtSetSessionThreadPoolSize(session_options, 1);
  // OrtSetSessionGraphOptimizationLevel(session_options, 1);

  OrtSession* session;
  // TODO: temporarily write modeldef to a file and load it back, until ORT implements it
  const char* model_path = "/tmp/net.onnx";
  onnx_status = OrtCreateSession(env, model_path, session_options, &session);

  if (onnx_status != NULL) {
    const char* msg = OrtGetErrorMessage(onnx_status);
    OrtReleaseStatus(onnx_status);
    // TODO fill err
    return NULL;
  }

  // OrtAllocator* allocator;
  // OrtCreateDefaultAllocator(&allocator);

  RAI_Model* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->model = NULL;
  ret->session = session;
  ret->backend = backend;
  ret->device = device;
  ret->refCount = 1;

  return ret;
}

void RAI_ModelFreeORT(RAI_Model* model, RAI_Error* error) {
  // TODO
}

int RAI_ModelRunORT(RAI_ModelRunCtx* mctx, RAI_Error *error) {
  // TODO

  return 0;
}

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
  // TODO

  return 0;
}
