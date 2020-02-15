#include <cuda_provider_factory.h>
#include "backends/onnxruntime.h"
#include "backends/util.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"

#include "onnxruntime_c_api.h"

int RAI_InitBackendORT(int (*get_api_fn)(const char *, void *)) {
  get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
  get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
  get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
  get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
  get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));

  return REDISMODULE_OK;
}

ONNXTensorElementDataType RAI_GetOrtDataTypeFromDL(DLDataType dtype) {
  if (dtype.code == kDLFloat) {
    switch (dtype.bits) {
      case 32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      case 64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
  else if (dtype.code == kDLInt) {
    switch (dtype.bits) {
      case 8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      case 16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
      case 32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
  else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
      case 8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      case 16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
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

OrtValue* RAI_OrtValueFromTensors(RAI_Tensor** ts, size_t count, RAI_Error *error) {
  OrtStatus* status = NULL;
  const OrtApi* ort = OrtGetApiBase()->GetApi(1);

  if (count == 0) {
    return NULL;
  }

  OrtAllocator *allocator;
  status = ort->GetAllocatorWithDefaultOptions(&allocator);
  if (status != NULL) {
    return NULL;
  }

  if (count == 0) {
    return NULL;
  }

  size_t batch_size = 0;
  size_t batch_byte_size = 0;

  for (size_t i=0; i<count; i++) {
    batch_size += ts[i]->tensor.dl_tensor.shape[0];
    batch_byte_size += RAI_TensorByteSize(ts[i]);
  }

  RAI_Tensor* t0 = ts[0];

  const int ndim = t0->tensor.dl_tensor.ndim;
  int64_t batched_shape[ndim];

  for (size_t i=1; i<ndim; i++) {
    batched_shape[i] = t0->tensor.dl_tensor.shape[i];
  }

  batched_shape[0] = batch_size;

  OrtValue* out;

  if (count > 1) {
    status = ort->CreateTensorAsOrtValue(
      allocator,
      batched_shape,
      t0->tensor.dl_tensor.ndim,
      RAI_GetOrtDataTypeFromDL(t0->tensor.dl_tensor.dtype),
      &out);
    if (status != NULL) {
      goto error;
    }
 
    char *ort_data;
    status = ort->GetTensorMutableData(out, (void **)&ort_data);
    if (status != NULL) {
      goto error;
    }

    size_t offset = 0;
    for (size_t i=0; i<count; i++) {
      memcpy(ort_data + offset, RAI_TensorData(ts[i]), RAI_TensorByteSize(ts[i]));
      offset += RAI_TensorByteSize(ts[i]);
    }
  }
  else {
   status = ort->CreateTensorWithDataAsOrtValue(
     allocator->Info(allocator),
     t0->tensor.dl_tensor.data,
     RAI_TensorByteSize(t0),
     t0->tensor.dl_tensor.shape,
     t0->tensor.dl_tensor.ndim,
     RAI_GetOrtDataTypeFromDL(t0->tensor.dl_tensor.dtype),
     &out);

    if (status != NULL) {
      goto error;
    }
  }

  return out;

error:
  RAI_SetError(error, RAI_EMODELCREATE, ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  return NULL;
}

RAI_Tensor* RAI_TensorCreateFromOrtValue(OrtValue* v, size_t batch_offset, size_t batch_size, RAI_Error *error) {
  OrtStatus* status = NULL;
  const OrtApi* ort = OrtGetApiBase()->GetApi(1);

  RAI_Tensor* ret = NULL;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
 
  int is_tensor;
  status = ort->IsTensor(v, &is_tensor);
  if (status != NULL) goto error;

  if (!is_tensor) {
    // TODO: if not tensor, flatten the data structure (sequence or map) and store it in a tensor.
    // If return value is string, emit warning.
    return NULL;
  }

  ret = RedisModule_Calloc(1, sizeof(*ret));

  DLContext ctx = (DLContext){
      .device_type = kDLCPU,
      .device_id = 0
  };

  OrtTensorTypeAndShapeInfo* info;
  status = ort->GetTensorTypeAndShape(v, &info);
  if (status != NULL) goto error;

  {
    size_t ndims;
    status = ort->GetDimensionsCount(info, &ndims);
    if (status != NULL) goto error;

    int64_t dims[ndims];
    status = ort->GetDimensions(info, dims, ndims);
    if (status != NULL) goto error;

    enum ONNXTensorElementDataType ort_dtype;
    status = ort->GetTensorElementType(info, &ort_dtype);
    if (status != NULL) goto error;

    int64_t total_batch_size = dims[0];

    shape = RedisModule_Calloc(ndims, sizeof(*shape));
    strides = RedisModule_Calloc(ndims, sizeof(*strides));
    for (int64_t i=0; i<ndims; ++i)
    {
      shape[i] = dims[i];
      strides[i] = 1;
    }
    shape[0] = batch_size;
    for (int64_t i = ndims - 2; i >= 0; --i)
    {
      strides[i] *= strides[i + 1] * shape[i + 1];
    }

    // size_t sample_bytesize = TF_TensorByteSize(tensor) / total_batch_size;

    DLDataType dtype = RAI_GetDLDataTypeFromORT(ort_dtype);
#ifdef RAI_COPY_RUN_OUTPUT
    char *ort_data;
    status = ort->GetTensorMutableData(v, (void **)&ort_data);
    if (status != NULL) {
      goto error;
    }
    size_t elem_count;
    status = ort->GetTensorShapeElementCount(info, &elem_count);
    if (status != NULL) {
      goto error;
    }

    const size_t len = dtype.bits * elem_count / 8;

    const size_t total_bytesize = len * sizeof(char);
    const size_t sample_bytesize = total_bytesize / total_batch_size;
    const size_t batch_bytesize = sample_bytesize * batch_size;

    char *data = RedisModule_Calloc(batch_bytesize, sizeof(*data));
    memcpy(data, ort_data + batch_offset * sample_bytesize, batch_bytesize);
#endif

    ort->ReleaseTensorTypeAndShapeInfo(info);

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
            .byte_offset = 0},
        .manager_ctx = NULL,
        .deleter = NULL};

    ret->refCount = 1;
    return ret;
  }

error:
  RAI_SetError(error, RAI_EMODELCREATE, ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  if (shape != NULL) {
    RedisModule_Free(shape);
  }
  if (strides != NULL) {
    RedisModule_Free(shape);
  }
  if (ret != NULL) {
    RedisModule_Free(ret);
  }
  return NULL;
}

typedef struct RAI_ONNXBuffer {
  char* data;
  size_t len;
} RAI_ONNXBuffer;

OrtEnv* env = NULL;

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, const char* devicestr, RAI_ModelOpts opts,
                              const char *modeldef, size_t modellen,
                              RAI_Error *error) {
  const OrtApi* ort = OrtGetApiBase()->GetApi(1);

  RAI_Device device;
  int64_t deviceid;

  if (!parseDeviceStr(devicestr, &device, &deviceid)) {
    RAI_SetError(error, RAI_EMODELCREATE, "ERR unsupported device");
    return NULL;
  }

  if (deviceid == -1){
    // ORT does not like device id as -1
    deviceid = 0;
  }

  OrtStatus* status = NULL;

  if (env == NULL) {
    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
  }

  if (status != NULL || env == NULL) {
    goto error;
  }

  // TODO: these options could be configured at the AI.CONFIG level
  OrtSessionOptions* session_options;
  status = ort->CreateSessionOptions(&session_options);
  if (status != NULL) {
    goto error;
  }

  // status = ort->SetSessionThreadPoolSize(session_options, 1);
  if (status != NULL) {
    ort->ReleaseSessionOptions(session_options);
    goto error;
  }

  status = ort->SetSessionGraphOptimizationLevel(session_options, 1);
  if (status != NULL) {
    ort->ReleaseSessionOptions(session_options);
    goto error;
  }
  // TODO: we will need to propose a more dynamic way to request a specific provider,
  // e.g. given the name, in ONNXRuntime
#if RAI_ONNXRUNTIME_USE_CUDA
  if (device == RAI_DEVICE_GPU) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, deviceid);
  }
#else
  // TODO: Do dynamic device/provider check with GetExecutionProviderType or something on these lines
  if (device == RAI_DEVICE_GPU) {
    RAI_SetError(error, RAI_EMODELCREATE, "GPU requested but ONNX couldn't find CUDA");
    return NULL;
  }
#endif

  OrtSession* session;

  status = ort->CreateSessionFromArray(env, modeldef, modellen, session_options, &session);

  ort->ReleaseSessionOptions(session_options);

  if (status != NULL) {
    goto error;
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
  ret->devicestr = RedisModule_Strdup(devicestr);
  ret->refCount = 1;
  ret->opts = opts;
  ret->data = onnxbuffer;

  return ret;

error:
  RAI_SetError(error, RAI_EMODELCREATE, ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  return NULL;
}

void RAI_ModelFreeORT(RAI_Model* model, RAI_Error* error) {
  const OrtApi* ort = OrtGetApiBase()->GetApi(1);

  RedisModule_Free(((RAI_ONNXBuffer*)(model->data))->data);
  RedisModule_Free(model->data);
  RedisModule_Free(model->devicestr);
  ort->ReleaseSession(model->session);

  model->model = NULL;
  model->session = NULL;
}

int RAI_ModelRunORT(RAI_ModelRunCtx *mctx, RAI_Error *error)
{
  const OrtApi* ort = OrtGetApiBase()->GetApi(1);

  OrtSession *session = mctx->model->session;

  if (session == NULL) {
    RAI_SetError(error, RAI_EMODELRUN, "ONNXRuntime session was not allocated\n");
    return 1;
  }

  const size_t nbatches = array_len(mctx->batches);
  if (nbatches == 0) {
    RAI_SetError(error, RAI_EMODELRUN, "No batches to run\n");
    return 1;
  }
 
  size_t batch_sizes[nbatches];
  size_t batch_offsets[nbatches];
  if (array_len(mctx->batches[0].inputs) > 0) {
    for (size_t b=0; b<nbatches; ++b) {
      batch_sizes[b] = RAI_TensorDim(mctx->batches[b].inputs[0].tensor, 0);
    }
    batch_offsets[0] = 0;
    for (size_t b=1; b<nbatches; ++b) {
      batch_offsets[b] = batch_offsets[b-1] + batch_sizes[b-1];
    }
  }

  OrtStatus *status = NULL;

  OrtAllocator *allocator;
  status = ort->GetAllocatorWithDefaultOptions(&allocator);
  if (status != NULL) {
    goto error;
  }

  size_t n_input_nodes;
  status = ort->SessionGetInputCount(session, &n_input_nodes);
  if (status != NULL) {
    goto error;
  }

  size_t n_output_nodes;
  status = ort->SessionGetOutputCount(session, &n_output_nodes);
  if (status != NULL) {
    goto error;
  }

  {
    const char *input_names[n_input_nodes];
    const char *output_names[n_output_nodes];

    OrtValue *inputs[n_input_nodes];
    OrtValue *outputs[n_output_nodes];

    const size_t ninputs = array_len(mctx->batches[0].inputs);
    const size_t noutputs = array_len(mctx->batches[0].outputs);

    if (ninputs != n_input_nodes) {
      char msg[70];
      sprintf(msg, "Expected %li inputs but got %li", n_input_nodes, ninputs);
      RAI_SetError(error, RAI_EMODELRUN, msg);
      return 1;
    }

    if (noutputs != n_output_nodes) {
      char msg[70];
      sprintf(msg, "Expected %li outputs but got %li", n_output_nodes, noutputs);
      RAI_SetError(error, RAI_EMODELRUN, msg);
      return 1;
    }

    for (size_t i = 0; i < n_input_nodes; i++) {
      char *input_name;
      status = ort->SessionGetInputName(session, i, allocator, &input_name);
      if (status != NULL) {
        goto error;
      }

      input_names[i] = input_name;

      RAI_Tensor* batched_input_tensors[nbatches];
      for (size_t b=0; b<nbatches; b++) {
        batched_input_tensors[b] = mctx->batches[b].inputs[i].tensor;
      }

      inputs[i] = RAI_OrtValueFromTensors(batched_input_tensors, nbatches, error);
      if (error->code != RAI_OK) {
        ort->ReleaseStatus(status);
        return 1;
      }

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

    for (size_t i = 0; i < n_output_nodes; i++) {
      char *output_name;
      status = ort->SessionGetOutputName(session, i, allocator, &output_name);
      if (status != NULL) {
        goto error;
      }

      output_names[i] = output_name;
      outputs[i] = NULL;
    }

    // ORT_API_STATUS(OrtRun, _Inout_ OrtSession* sess,
    //                _In_ OrtRunOptions* run_options,
    //                _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
    //                _In_ const char* const* output_names, size_t output_names_len, _Out_ OrtValue** output);
    OrtRunOptions *run_options = NULL;
    status = ort->Run(session, run_options, input_names, (const OrtValue *const *)inputs,
                     n_input_nodes, output_names, n_output_nodes, outputs);

    if (status) {
      goto error;
    }

    for (size_t i = 0; i < n_output_nodes; i++) {
      for (size_t b=0; b<nbatches; b++) {
        RAI_Tensor* output_tensor = RAI_TensorCreateFromOrtValue(outputs[i], batch_offsets[b], batch_sizes[b], error);
        if (error->code != RAI_OK) {
          ort->ReleaseStatus(status);
          return 1;
        }
        if (output_tensor) {
          mctx->batches[b].outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
          RAI_TensorFree(output_tensor);
        }
        else {
          printf("ERR: non-tensor output from ONNX models, ignoring (currently unsupported).\n");
        }
      }

      ort->ReleaseValue(outputs[i]);
    }

    for (size_t i = 0; i < n_input_nodes; i++) {
      ort->ReleaseValue(inputs[i]);
    }

    return 0;
  }

error:
  RAI_SetError(error, RAI_EMODELRUN, ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  return 1;
}

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
  RAI_ONNXBuffer* onnxbuffer = (RAI_ONNXBuffer*)model->data;
  *buffer = RedisModule_Calloc(onnxbuffer->len, sizeof(char));
  memcpy(*buffer, onnxbuffer->data, onnxbuffer->len);
  *len = onnxbuffer->len;

  return 0;
}
