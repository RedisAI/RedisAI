#include "backends/onnxruntime.h"
#include "tensor.h"
#include "util/alloc.h"
#include "util/arr_rm_alloc.h"

// TODO for getpid, remove when ONNXRuntime can read from stream
#include <sys/types.h>
#include <unistd.h>

ONNXTensorElementDataType RAI_GetOrtDataTypeFromDL(DLDataType dtype) {
  if (dtype.code == kDLFloat) {
    switch (dtype.bits) {
      case 32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; break;
      case 64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; break;
      default:
        return 0;
    }
  }
  else if (dtype.code == kDLInt) {
    switch (dtype.bits) {
      case 8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; break;
      case 16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16; break;
      case 32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; break;
      case 64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; break;
      default:
        return 0;
    }
  }
  else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
      case 8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; break;
      case 16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16; break;
      default:
        return 0;
    }
  }
  return 0;
}

DLDataType RAI_GetDLDataTypeFromORT(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return (DLDataType){ .code = kDLFloat, .bits = 32, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return (DLDataType){ .code = kDLFloat, .bits = 64, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return (DLDataType){ .code = kDLInt, .bits = 8, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return (DLDataType){ .code = kDLInt, .bits = 16, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return (DLDataType){ .code = kDLInt, .bits = 32, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return (DLDataType){ .code = kDLInt, .bits = 64, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return (DLDataType){ .code = kDLUInt, .bits = 8, .lanes = 1 };
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return (DLDataType){ .code = kDLUInt, .bits = 16, .lanes = 1 };
    default:
      return (DLDataType){ .bits = 0 };
  }
  return (DLDataType){ .bits = 0 };
}

OrtValue* RAI_OrtValueFromTensor(RAI_Tensor* t) {
  // TODO: create outside and pass?
  OrtAllocatorInfo* allocator_info;
  OrtStatus* status;
  status = OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info);

  OrtValue* out;
  status = OrtCreateTensorWithDataAsOrtValue(
    allocator_info,
    t->tensor.dl_tensor.data,
    RAI_TensorByteSize(t),
    t->tensor.dl_tensor.shape,
    t->tensor.dl_tensor.ndim,
    RAI_GetOrtDataTypeFromDL(t->tensor.dl_tensor.dtype),
    &out);

  OrtReleaseAllocatorInfo(allocator_info);

  return out;
}

RAI_Tensor* RAI_TensorCreateFromOrtValue(OrtValue* v) {
  RAI_Tensor* ret = RedisModule_Calloc(1, sizeof(*ret));

  DLContext ctx = (DLContext){
      .device_type = kDLCPU,
      .device_id = 0
  };

  OrtStatus* status = NULL;

  OrtTensorTypeAndShapeInfo* info = OrtCreateTensorTypeAndShapeInfo();
  status = OrtGetTensorShapeAndType(v, &info);
  // TODO: deal with status

  size_t ndims = OrtGetNumOfDimensions(info);

  int64_t dims[ndims];
  OrtGetDimensions(info, dims, ndims);

  enum ONNXTensorElementDataType ort_dtype = OrtGetTensorElementType(info);

  OrtReleaseTensorTypeAndShapeInfo(info);

  int64_t* shape = RedisModule_Calloc(ndims, sizeof(*shape));
  int64_t* strides = RedisModule_Calloc(ndims, sizeof(*strides));
  for (int64_t i = 0 ; i < ndims ; ++i) {
    shape[i] = dims[i];
    strides[i] = 1;
  }
  for (int64_t i = ndims-2 ; i >= 0 ; --i) {
    strides[i] *= strides[i+1] * shape[i+1];
  }

  DLDataType dtype = RAI_GetDLDataTypeFromORT(ort_dtype);
#ifdef RAI_COPY_RUN_OUTPUT
  char* ort_data;
  status = OrtGetTensorMutableData(v, (void**)&ort_data);
  size_t len = dtype.bits * OrtGetTensorShapeElementCount(info);
  char* data = RedisModule_Calloc(len, sizeof(*data));
  memcpy(data, ort_data, len);
#endif

  // TODO: use manager_ctx to ensure ORT tensor doesn't get deallocated
  // This applies to outputs

  ret->tensor = (DLManagedTensor){
    .dl_tensor = (DLTensor){
      .ctx = ctx,
#ifdef RAI_COPY_RUN_OUTPUT
      .data = data,
#else
      #error zero-copy passing output memory from ORT not currently supported
#endif
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

typedef struct RAI_ONNXBuffer {
  char* data;
  size_t len;
} RAI_ONNXBuffer;

OrtEnv* env = NULL;

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, RAI_Device device,
                              const char *modeldef, size_t modellen,
                              RAI_Error *error) {

  // TODO: take from
  // https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

  OrtStatus* onnx_status = NULL;

  if (env == NULL) {
    onnx_status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
  }

  if (onnx_status != NULL) {
    const char* msg = OrtGetErrorMessage(onnx_status);
    RAI_SetError(error, RAI_EMODELRUN, msg);
    OrtReleaseStatus(onnx_status);
    return NULL;
  }

  // TODO: probably these options could be configured at the AI.CONFIG level
  OrtSessionOptions* session_options = OrtCreateSessionOptions();
  OrtSetSessionThreadPoolSize(session_options, 1);
  OrtSetSessionGraphOptimizationLevel(session_options, 1);

  // TODO: we will need to propose a more dynamic way to request a specific provider,
  // e.g. given the name, in ONNXRuntiem
#if RAI_ONNXRUNTIME_USE_CUDA
  if (device == RAI_DEVICE_GPU) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
  }
#endif

  OrtSession* session;

#define RAI_ONNX_NO_SESSION_FROM_ARRAY
#ifdef RAI_ONNX_NO_SESSION_FROM_ARRAY
  // NOTE This works as long as setting a model is mono-thread
  // Since this solution is temporary until ONNXRuntime implements
  // loading a model from a buffer, we'll keep this as is
  char tmpfile[256];
  FILE *fp;
  snprintf(tmpfile,256,"temp-%d.onnx", (int) getpid());
  fp = fopen(tmpfile,"w");
  if (!fp) {
    RAI_SetError(error, RAI_EMODELCREATE, "ERR: cannot write ONNX tmp file\n");
    return NULL;
  }
  size_t nwritten = fwrite(modeldef, sizeof(char), modellen, fp);
  if (nwritten < modellen ||
      fflush(fp) == EOF || 
      fsync(fileno(fp)) == -1 ||
      fclose(fp) == EOF) {
    RAI_SetError(error, RAI_EMODELCREATE, "ERR: cannot flush or close ONNX tmp file\n");
    return NULL;
  }
  onnx_status = OrtCreateSession(env, tmpfile, session_options, &session);
#else
  onnx_status = OrtCreateSessionFromArray(env, modeldef, modellen, session_options, &session);
#endif

  OrtReleaseSessionOptions(session_options);

#ifdef RAI_ONNX_NO_SESSION_FROM_ARRAY
  unlink(tmpfile);
#endif

  if (onnx_status != NULL) {
    const char* msg = OrtGetErrorMessage(onnx_status);
    RAI_SetError(error, RAI_EMODELCREATE, msg);
    OrtReleaseStatus(onnx_status);
    return NULL;
  }

  // Since ONNXRuntime doesn't have a re-serialization function,
  // we cache the blob in order to re-serialize it.
  // Not optimal for storage purposes, but again, it may be temporary
  char* buffer = RedisModule_Calloc(modellen, sizeof(*buffer));
  memcpy(buffer, modeldef, modellen);

  RAI_ONNXBuffer* onnxbuffer = RedisModule_Calloc(1, sizeof(*onnxbuffer));
  onnxbuffer->data = buffer;
  onnxbuffer->len = modellen;

  RAI_Model* ret = RedisModule_Calloc(1, sizeof(*ret));
  ret->model = NULL;
  ret->session = session;
  ret->backend = backend;
  ret->device = device;
  ret->refCount = 1;
  ret->data = onnxbuffer;

  return ret;
}

void RAI_ModelFreeORT(RAI_Model* model, RAI_Error* error) {
  RedisModule_Free(((RAI_ONNXBuffer*)(model->data))->data);
  RedisModule_Free(model->data);
  OrtReleaseSession(model->session);

  model->model = NULL;
  model->session = NULL;
}

int RAI_ModelRunORT(RAI_ModelRunCtx* mctx, RAI_Error *error) {
  OrtSession* session = mctx->model->session;

  OrtStatus* status;

  OrtAllocator* allocator;
  status = OrtCreateDefaultAllocator(&allocator);

  size_t n_input_nodes;
  status = OrtSessionGetInputCount(session, &n_input_nodes);

  size_t n_output_nodes;
  status = OrtSessionGetOutputCount(session, &n_output_nodes);

  const char* input_names[n_input_nodes];
  const char* output_names[n_output_nodes];

  OrtValue* inputs[n_input_nodes];
  OrtValue* outputs[n_output_nodes];

  size_t ninputs = array_len(mctx->inputs);
  size_t noutputs = array_len(mctx->outputs);

  if (ninputs != n_input_nodes) {
    RAI_SetError(error, RAI_EMODELRUN, "Unexpected number of inputs for graph\n");
    return 1;
  }

  if (noutputs != n_output_nodes) {
    RAI_SetError(error, RAI_EMODELRUN, "Unexpected number of outputs for graph\n");
    return 1;
  }

  for (size_t i=0; i<n_input_nodes; i++) {
    char* input_name;
    status = OrtSessionGetInputName(session, i, allocator, &input_name);
    input_names[i] = input_name;

    inputs[i] = RAI_OrtValueFromTensor(mctx->inputs[i].tensor);

    // TODO: use this to check input dim, shapes
    #if 0
    OrtTypeInfo* typeinfo;
    status = OrtSessionGetInputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
    ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
    // printf("Input %zu : type=%d\n", i, type);

    size_t num_dims = OrtGetDimensionsCount(tensor_info);
    // printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims.resize(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++) {
      // printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
    }

    OrtReleaseTypeInfo(typeinfo);
    #endif
  }

  for (size_t i=0; i<n_output_nodes; i++) {
    char* output_name;
    status = OrtSessionGetOutputName(session, i, allocator, &output_name);
    output_names[i] = output_name;
  }

  OrtReleaseAllocator(allocator);

  // ORT_API_STATUS(OrtRun, _Inout_ OrtSession* sess,
  //                _In_ OrtRunOptions* run_options,
  //                _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
  //                _In_ const char* const* output_names, size_t output_names_len, _Out_ OrtValue** output);
  OrtRunOptions* run_options = NULL;
  status = OrtRun(session, run_options, input_names, (const OrtValue* const*) inputs,
                  n_input_nodes, output_names, n_output_nodes, outputs);

  if (status) {
    const char* msg = OrtGetErrorMessage(status);
    RAI_SetError(error, RAI_EMODELRUN, msg);
    OrtReleaseStatus(status);
    return 1;
  }

  for (size_t i=0; i<n_output_nodes; i++) {
    RAI_Tensor* output_tensor = RAI_TensorCreateFromOrtValue(outputs[i]);
    mctx->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
    RAI_TensorFree(output_tensor);
 
    OrtReleaseValue(outputs[i]);
  }

  for (size_t i=0; i<n_input_nodes; i++) {
    OrtReleaseValue(inputs[i]);
  }

  return 0;
}

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
  RAI_ONNXBuffer* onnxbuffer = (RAI_ONNXBuffer*)model->data;
  *buffer = RedisModule_Calloc(onnxbuffer->len, sizeof(char));
  memcpy(*buffer, onnxbuffer->data, onnxbuffer->len);

  return 0;
}
