#include "backends/tensorflow.h"
#include "tensor.h"
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

  long long* shape = RedisModule_Alloc(ndims * sizeof(*shape));
  for (long long i = 0 ; i < ndims ; ++i){
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


RAI_Graph *RAI_GraphCreateTF(RAI_Backend backend, RAI_Device device,
                             const char *graphdef, size_t graphlen) {
  TF_Graph* graph = TF_NewGraph();

  TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();

  TF_Buffer *buffer = TF_NewBuffer();
  buffer->length = graphlen;
  buffer->data = graphdef;

  TF_Status *status = TF_NewStatus();

  TF_GraphImportGraphDef(graph, buffer, options, status);

  if (TF_GetCode(status) != TF_OK) {
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
    // TODO: free memory
    return NULL;
  }
  TF_DeleteStatus(optionsStatus);

  TF_Status *sessionStatus = TF_NewStatus();
  TF_Session *session = TF_NewSession(graph, sessionOptions, sessionStatus);

  if (TF_GetCode(sessionStatus) != TF_OK) {
    // TODO: free memory
    return NULL;
  }

  TF_DeleteSessionOptions(sessionOptions);
  TF_DeleteStatus(sessionStatus);

  RAI_Graph* ret = RedisModule_Alloc(sizeof(*ret));
  ret->graph = graph;
  ret->session = session;
  ret->backend = backend;
  ret->refCount = 1;

  return ret;
}

void RAI_GraphFreeTF(RAI_Graph* graph) {
  TF_Status *status = TF_NewStatus();
  TF_CloseSession(graph->session, status);

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteSession(graph->session, status);
  graph->session = NULL;

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteGraph(graph->graph);
  graph->graph = NULL;

  TF_DeleteStatus(status);
}

int RAI_GraphRunTF(RAI_GraphRunCtx* gctx) {
  TF_Status *status = TF_NewStatus();

  TF_Tensor* inputTensorsValues[array_len(gctx->inputs)];
  TF_Output inputs[array_len(gctx->inputs)];
  TF_Tensor* outputTensorsValues[array_len(gctx->outputs)];
  TF_Output outputs[array_len(gctx->outputs)];

  for (size_t i = 0 ; i < array_len(gctx->inputs); ++i) {
    inputTensorsValues[i] = RAI_TFTensorFromTensor(gctx->inputs[i].tensor);
    TF_Output port;
    port.oper = TF_GraphOperationByName(gctx->graph->graph, gctx->inputs[i].name);
    port.index = 0;
    if(port.oper == NULL){
      return 0;
    }
    inputs[i] = port;
  }

  for (size_t i = 0 ; i < array_len(gctx->outputs) ; ++i) {
    TF_Output port;
    port.oper = TF_GraphOperationByName(gctx->graph->graph, gctx->outputs[i].name);
    port.index = 0;
    if(port.oper == NULL){
      return 0;
    }
    outputs[i] = port;
  }

  TF_SessionRun(gctx->graph->session, NULL /* run_options */,
                inputs, inputTensorsValues, array_len(gctx->inputs),
                outputs, outputTensorsValues, array_len(gctx->outputs),
                NULL /* target_opers */, 0 /* ntargets */,
                NULL /* run_Metadata */,
                status);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    return 0;
  }

  for(size_t i = 0 ; i < array_len(gctx->outputs) ; ++i) {
    RAI_Tensor* output_tensor = RAI_TensorCreateFromTFTensor(outputTensorsValues[i]);
    gctx->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
  }

  TF_DeleteStatus(status);

  return 1;
}
