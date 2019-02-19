#include "backends/tensorflow.h"
#include "tensor.h"
#include "utils/alloc.h"
#include "utils/arr_rm_alloc.h"

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
  RAI_Tensor* ret = RedisModule_Alloc(sizeof(*ret));

  DLContext ctx = (DLContext){
      .device_type = kDLCPU,
      .device_id = 0
  };

  size_t ndims = TF_NumDims(tensor);

  int64_t* shape = RedisModule_Alloc(ndims * sizeof(*shape));
  for (long i = 0 ; i < ndims ; ++i){
    shape[i] = TF_Dim(tensor, i);
  }

  // FIXME: In TF, RunSession allocates memory for output tensors
  // This means that we either memcpy the tensor data and let
  // Redis be responsible for the memory, or we reuse the TF
  // allocated memory, which might not be optimal down the road
  // Note: on YOLO this has no impact on perf
#ifdef RAI_COPY_RUN_OUTPUT
  size_t len = TF_TensorByteSize(tensor);
  char* data = RedisModule_Alloc(len * sizeof(*data));
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
      .strides = NULL,
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


RAI_Model *RAI_ModelCreateTF(RAI_Backend backend, RAI_Device device,
                             char **inputs, char **outputs,
                             const char *modeldef, size_t modellen) {
  TF_Graph* model = TF_NewGraph();

  TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();

  TF_Buffer *buffer = TF_NewBuffer();
  buffer->length = modellen;
  buffer->data = modeldef;

  TF_Status *status = TF_NewStatus();

  TF_GraphImportGraphDef(model, buffer, options, status);

  if (TF_GetCode(status) != TF_OK) {
    printf("ERR: Failed importing TF graph\n");
    // todo: free memory
    return NULL;
  }

  TF_DeleteImportGraphDefOptions(options);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(status);

  TF_Status *optionsStatus = TF_NewStatus();

  TF_SessionOptions *sessionOptions = TF_NewSessionOptions();

  // For setting config options in session from the C API see:
  // https://github.com/tensorflow/tensorflow/issues/13853
  // import tensorflow as tf
  // config = tf.ConfigProto(device_count = {'GPU': 0})
  // serialized = config.SerializeToString()
  // result = list(map(hex, serialized))
  // print(result)

  // TODO: complain if device is GPU and GPU not available?
  if (device == RAI_DEVICE_CPU) {
    uint8_t config[9] = {0x0a, 0x07, 0x0a, 0x03, 0x47, 0x50, 0x55, 0x10, 0x00};
    TF_SetConfig(sessionOptions, (void *)config, 9, status);
  }

  if (TF_GetCode(optionsStatus) != TF_OK) {
    printf("ERR: Failed setting TF config\n");
    // TODO: free memory
    return NULL;
  }
  TF_DeleteStatus(optionsStatus);

  TF_Status *sessionStatus = TF_NewStatus();
  TF_Session *session = TF_NewSession(model, sessionOptions, sessionStatus);

  if (TF_GetCode(sessionStatus) != TF_OK) {
    printf("ERR: Failed creating TF session\n");
    // TODO: free memory
    return NULL;
  }

  TF_DeleteSessionOptions(sessionOptions);
  TF_DeleteStatus(sessionStatus);

  size_t ninputs = array_len(inputs);
  char **inputs_ = array_new(char*, ninputs);
  for (long long i=0; i<ninputs; i++) {
    char* input = RedisModule_Alloc(strlen(inputs[i]) * sizeof(char));
    strcpy(input, inputs[i]);
    array_append(inputs_, input);
  }

  size_t noutputs = array_len(outputs);
  char **outputs_ = array_new(char*, noutputs);
  for (long long i=0; i<noutputs; i++) {
    char* output = RedisModule_Alloc(strlen(outputs[i]) * sizeof(char));
    strcpy(output, outputs[i]);
    array_append(outputs_, output);
  }

  RAI_Model* ret = RedisModule_Alloc(sizeof(*ret));
  ret->model = model;
  ret->session = session;
  ret->backend = backend;
  ret->inputs = inputs_;
  ret->outputs = outputs_;
  ret->refCount = 1;

  return ret;
}

void RAI_ModelFreeTF(RAI_Model* model) {
  TF_Status *status = TF_NewStatus();
  TF_CloseSession(model->session, status);

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteSession(model->session, status);
  model->session = NULL;

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteGraph(model->model);
  model->model = NULL;

  if (model->inputs) {
    size_t ninputs = array_len(model->inputs);
    for (long long i=0; i<ninputs; i++) {
      RedisModule_Free(model->inputs[i]);
    }
    array_free(model->inputs);
  }

  if (model->outputs) {
    size_t noutputs = array_len(model->outputs);
    for (long long i=0; i<noutputs; i++) {
      RedisModule_Free(model->outputs[i]);
    }
    array_free(model->outputs);
  }

  TF_DeleteStatus(status);
}

int RAI_ModelRunTF(RAI_ModelRunCtx* mctx) {
  TF_Status *status = TF_NewStatus();

  TF_Tensor* inputTensorsValues[array_len(mctx->inputs)];
  TF_Output inputs[array_len(mctx->inputs)];
  TF_Tensor* outputTensorsValues[array_len(mctx->outputs)];
  TF_Output outputs[array_len(mctx->outputs)];

  for (size_t i=0 ; i<array_len(mctx->inputs); ++i) {
    inputTensorsValues[i] = RAI_TFTensorFromTensor(mctx->inputs[i].tensor);
    TF_Output port;
    port.oper = TF_GraphOperationByName(mctx->model->model, mctx->inputs[i].name);
    port.index = 0;
    if(port.oper == NULL){
      return 0;
    }
    inputs[i] = port;
  }

  for (size_t i=0 ; i<array_len(mctx->outputs) ; ++i) {
    TF_Output port;
    port.oper = TF_GraphOperationByName(mctx->model->model, mctx->outputs[i].name);
    port.index = 0;
    if(port.oper == NULL){
      return 0;
    }
    outputs[i] = port;
  }

  TF_SessionRun(mctx->model->session, NULL /* run_options */,
                inputs, inputTensorsValues, array_len(mctx->inputs),
                outputs, outputTensorsValues, array_len(mctx->outputs),
                NULL /* target_opers */, 0 /* ntargets */,
                NULL /* run_Metadata */,
                status);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    return 0;
  }

  for(size_t i = 0 ; i < array_len(mctx->outputs) ; ++i) {
    RAI_Tensor* output_tensor = RAI_TensorCreateFromTFTensor(outputTensorsValues[i]);
    mctx->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
  }

  TF_DeleteStatus(status);

  return 1;
}
