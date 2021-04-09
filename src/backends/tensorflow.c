#define REDISMODULE_MAIN
#include "backends/util.h"
#include "backends/tensorflow.h"
#include "util/arr.h"
#include "redis_ai_objects/model.h"
#include "redis_ai_objects/tensor.h"

#include "tensorflow/c/c_api.h"

int RAI_InitBackendTF(int (*get_api_fn)(const char *, void *)) {
    get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
    get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
    get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
    get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
    get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));

    return REDISMODULE_OK;
}

TF_DataType RAI_GetTFDataTypeFromDL(DLDataType dtype) {

    if (dtype.code == kDLFloat) {
        switch (dtype.bits) {
        case 32:
            return TF_FLOAT;
            break;
        case 64:
            return TF_DOUBLE;
            break;
        default:
            return 0;
        }
    } else if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        case 8:
            return TF_INT8;
            break;
        case 16:
            return TF_INT16;
            break;
        case 32:
            return TF_INT32;
            break;
        case 64:
            return TF_INT64;
            break;
        default:
            return 0;
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            return TF_UINT8;
            break;
        case 16:
            return TF_UINT16;
            break;
        default:
            return 0;
        }
    }
    return 0;
}

DLDataType RAI_GetDLDataTypeFromTF(TF_DataType dtype) {
    switch (dtype) {
    case TF_FLOAT:
        return (DLDataType){.code = kDLFloat, .bits = 32, .lanes = 1};
    case TF_DOUBLE:
        return (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1};
    case TF_INT8:
        return (DLDataType){.code = kDLInt, .bits = 8, .lanes = 1};
    case TF_INT16:
        return (DLDataType){.code = kDLInt, .bits = 16, .lanes = 1};
    case TF_INT32:
        return (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1};
    case TF_INT64:
        return (DLDataType){.code = kDLInt, .bits = 64, .lanes = 1};
    case TF_UINT8:
        return (DLDataType){.code = kDLUInt, .bits = 8, .lanes = 1};
    case TF_UINT16:
        return (DLDataType){.code = kDLUInt, .bits = 16, .lanes = 1};
    default:
        return (DLDataType){.bits = 0};
    }
    return (DLDataType){.bits = 0};
}

RAI_Tensor *RAI_TensorCreateFromTFTensor(TF_Tensor *tensor, size_t batch_offset,
                                         long long batch_size) {
    RAI_Tensor *ret = RAI_TensorNew();

    DLDevice device = (DLDevice){.device_type = kDLCPU, .device_id = 0};

    const size_t ndims = TF_NumDims(tensor);

    int64_t total_batch_size = TF_Dim(tensor, 0);
    total_batch_size = total_batch_size > 0 ? total_batch_size : 1;

    int64_t *shape = RedisModule_Calloc(ndims, sizeof(*shape));
    int64_t *strides = RedisModule_Calloc(ndims, sizeof(*strides));
    for (int64_t i = 0; i < ndims; ++i) {
        shape[i] = TF_Dim(tensor, i);
        strides[i] = 1;
    }
    if (batch_size != -1) {
        shape[0] = batch_size;
    } else {
        batch_size = total_batch_size;
    }
    for (int64_t i = ndims - 2; i >= 0; --i) {
        strides[i] *= strides[i + 1] * shape[i + 1];
    }

    const size_t sample_bytesize = TF_TensorByteSize(tensor) / total_batch_size;

    // FIXME: In TF, RunSession allocates memory for output tensors
    // This means that we either memcpy the tensor data and let
    // Redis be responsible for the memory, or we reuse the TF
    // allocated memory, which might not be optimal down the road
    // Note: on YOLO this has no impact on perf
#ifdef RAI_COPY_RUN_OUTPUT
    const size_t len = sample_bytesize * batch_size;
    char *data = RedisModule_Calloc(len, sizeof(*data));
    memcpy(data, TF_TensorData(tensor) + sample_bytesize * batch_offset, len);
#endif

    // TODO: use manager_ctx to ensure TF tensor doesn't get deallocated
    // This applies to outputs

    ret->tensor = (DLManagedTensor){
        .dl_tensor = (DLTensor){.device = device,
#ifdef RAI_COPY_RUN_OUTPUT
                                .data = data,
#else
                                .data = TF_TensorData(tensor),
#endif
                                .ndim = ndims,
                                .dtype = RAI_GetDLDataTypeFromTF(TF_TensorType(tensor)),
                                .shape = shape,
                                .strides = strides,
                                .byte_offset = 0},
        .manager_ctx = NULL,
        .deleter = NULL};

    return ret;
}

void RAI_TFDeallocator(void *data, size_t len, void *arg) {
    // printf("DEALLOCATOR CALLED\n");
    // do nothing, memory is managed by Redis
}

TF_Tensor *RAI_TFTensorFromTensor(RAI_Tensor *t) {
#ifdef RAI_COPY_RUN_INPUT
    TF_Tensor *out = TF_AllocateTensor(RAI_GetTFDataTypeFromDL(t->tensor.dl_tensor.dtype),
                                       t->tensor.dl_tensor.shape, t->tensor.dl_tensor.ndim,
                                       RAI_TensorByteSize(t));
    memcpy(TF_TensorData(out), t->tensor.dl_tensor.data, TF_TensorByteSize(out));
    return out;
#else
    return TF_NewTensor(RAI_GetTFDataTypeFromDL(t->tensor.dl_tensor.dtype),
                        t->tensor.dl_tensor.shape, t->tensor.dl_tensor.ndim,
                        t->tensor.dl_tensor.data, RAI_TensorByteSize(t), &RAI_TFDeallocator, NULL);
#endif /* RAI_COPY_RUN_INPUT */
}

TF_Tensor *RAI_TFTensorFromTensors(RAI_Tensor **ts, size_t count) {

    if (count == 0) {
        return NULL;
    }

    size_t batch_size = 0;
    size_t batch_byte_size = 0;

    for (size_t i = 0; i < count; i++) {
        batch_size += ts[i]->tensor.dl_tensor.shape[0];
        batch_byte_size += RAI_TensorByteSize(ts[i]);
    }

    RAI_Tensor *t0 = ts[0];

    const int ndim = t0->tensor.dl_tensor.ndim;
    int64_t batched_shape[ndim];

    for (size_t i = 0; i < ndim; i++) {
        batched_shape[i] = t0->tensor.dl_tensor.shape[i];
    }

    batched_shape[0] = batch_size;

    TF_Tensor *out = NULL;

    if (count > 1) {
        out = TF_AllocateTensor(RAI_GetTFDataTypeFromDL(t0->tensor.dl_tensor.dtype), batched_shape,
                                t0->tensor.dl_tensor.ndim, batch_byte_size);

        size_t offset = 0;
        for (size_t i = 0; i < count; i++) {
            size_t tbytesize = RAI_TensorByteSize(ts[i]);
            memcpy(TF_TensorData(out) + offset, ts[i]->tensor.dl_tensor.data, tbytesize);
            offset += tbytesize;
        }
    } else {
        out = TF_NewTensor(RAI_GetTFDataTypeFromDL(t0->tensor.dl_tensor.dtype),
                           t0->tensor.dl_tensor.shape, t0->tensor.dl_tensor.ndim,
                           t0->tensor.dl_tensor.data, RAI_TensorByteSize(t0), &RAI_TFDeallocator,
                           NULL);
    }

    return out;
}

int RAI_ModelCreateTF(RAI_Model *model, RAI_Error *error) {
    RAI_Device device;
    int64_t deviceid;

    if (!parseDeviceStr(model->devicestr, &device, &deviceid)) {
        RAI_SetError(error, RAI_EMODELIMPORT, "ERR unsupported device");
    }

    TF_Graph *graph = TF_NewGraph();
    TF_Status *status = TF_NewStatus();
    TF_Buffer *tfbuffer = TF_NewBuffer();
    TF_ImportGraphDefOptions *options = TF_NewImportGraphDefOptions();
    TF_Status *optionsStatus = NULL;
    TF_SessionOptions *sessionOptions = NULL;
    TF_Status *sessionStatus = NULL;
    TF_Session *session = NULL;

    tfbuffer->length = model->datalen;
    tfbuffer->data = model->data;

    TF_GraphImportGraphDef(graph, tfbuffer, options, status);

    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELIMPORT, errorMessage);
        RedisModule_Free(errorMessage);
        return REDISMODULE_ERR;
    }

    // Validate that the given inputs and outputs exist in the graph.
    for (size_t i = 0; i < model->ninputs; ++i) {
        TF_Operation *oper = TF_GraphOperationByName(graph, model->inputs[i]);
        if (oper == NULL || strcmp(TF_OperationOpType(oper), "Placeholder") != 0) {
            size_t len = strlen(model->inputs[i]);
            char *msg = RedisModule_Calloc(60 + len, sizeof(*msg));
            sprintf(msg, "ERR Input node named \"%s\" not found in TF graph.", model->inputs[i]);
            RAI_SetError(error, RAI_EMODELIMPORT, msg);
            RedisModule_Free(msg);
            goto cleanup;
        }
    }

    for (size_t i = 0; i < model->noutputs; ++i) {
        TF_Operation *oper = TF_GraphOperationByName(graph, model->outputs[i]);
        if (oper == NULL) {
            size_t len = strlen(model->outputs[i]);
            char *msg = RedisModule_Calloc(60 + len, sizeof(*msg));
            sprintf(msg, "ERR Output node named \"%s\" not found in TF graph", model->outputs[i]);
            RAI_SetError(error, RAI_EMODELIMPORT, msg);
            RedisModule_Free(msg);
            goto cleanup;
        }
    }

    TF_DeleteImportGraphDefOptions(options);
    options = NULL;
    TF_DeleteBuffer(tfbuffer);
    tfbuffer = NULL;
    TF_DeleteStatus(status);
    status = NULL;

    optionsStatus = TF_NewStatus();
    sessionOptions = TF_NewSessionOptions();

    // For setting config options in session from the C API see:
    // https://github.com/tensorflow/tensorflow/issues/13853
    // import tensorflow as tf
    // config = tf.ConfigProto(device_count = {'GPU': 0})
    // serialized = config.SerializeToString()
    // result = list(map(hex, serialized))
    // print(result)

    if (device == RAI_DEVICE_CPU) {
        // Set number of GPU to 0 with
        // config.device_count = {'GPU': 0}
        uint8_t config[] = {0x0a, 0x07, 0x0a, 0x03, 0x47, 0x50, 0x55, 0x10, 0x00};
        TF_SetConfig(sessionOptions, (void *)config, sizeof(config), optionsStatus);

        if (TF_GetCode(optionsStatus) != TF_OK) {
            RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(optionsStatus)));
            goto cleanup;
        }

        RAI_ModelOpts opts = model->opts;
        if (opts.backends_intra_op_parallelism > 0) {
            uint8_t proto[] = {0x10, (uint8_t)opts.backends_intra_op_parallelism};
            TF_SetConfig(sessionOptions, proto, sizeof(proto), optionsStatus);
            if (TF_GetCode(optionsStatus) != TF_OK) {
                RAI_SetError(error, RAI_EMODELCONFIGURE,
                             RedisModule_Strdup(TF_Message(optionsStatus)));
                goto cleanup;
            }
        }

        if (opts.backends_inter_op_parallelism > 0) {
            uint8_t proto1[] = {0x28, (uint8_t)opts.backends_inter_op_parallelism};
            TF_SetConfig(sessionOptions, proto1, sizeof(proto1), optionsStatus);
            if (TF_GetCode(optionsStatus) != TF_OK) {
                RAI_SetError(error, RAI_EMODELCONFIGURE,
                             RedisModule_Strdup(TF_Message(optionsStatus)));
                goto cleanup;
            }
        }
    } else if (device == RAI_DEVICE_GPU) {
        if (deviceid == -1) {
            // Set
            // config.gpu_options.allow_growth = True
            uint8_t config[4] = {0x32, 0x02, 0x20, 0x01};
            TF_SetConfig(sessionOptions, (void *)config, 4, optionsStatus);
        } else {
            // Set
            // config.gpu_options.allow_growth = True
            // config.gpu_options.visible_device_list = '<deviceid>'
            uint8_t config[7] = {0x32, 0x05, 0x20, 0x01, 0x2a, 0x01, 0x30};
            config[6] += (uint8_t)deviceid;
            TF_SetConfig(sessionOptions, (void *)config, 7, optionsStatus);
        }
    }

    if (TF_GetCode(optionsStatus) != TF_OK) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(optionsStatus)));
        goto cleanup;
    }
    TF_DeleteStatus(optionsStatus);
    optionsStatus = NULL;
    sessionStatus = TF_NewStatus();
    session = TF_NewSession(graph, sessionOptions, sessionStatus);

    TF_Status *deviceListStatus = TF_NewStatus();
    TF_DeviceList *deviceList = TF_SessionListDevices(session, deviceListStatus);
    const int num_devices = TF_DeviceListCount(deviceList);
    bool foundNoGPU = true;
    for (int i = 0; i < num_devices; ++i) {
        const char *device_type = TF_DeviceListType(deviceList, i, deviceListStatus);
        bool cmp = strcmp(device_type, "GPU");
        if (!cmp) {
            foundNoGPU = false;
            break;
        }
    }
    if (foundNoGPU && device == RAI_DEVICE_GPU) {
        RAI_SetError(error, RAI_EMODELCREATE, "ERR GPU requested but TF couldn't find CUDA");
        TF_DeleteDeviceList(deviceList);
        TF_DeleteStatus(deviceListStatus);
        goto cleanup;
    }
    TF_DeleteDeviceList(deviceList);
    TF_DeleteStatus(deviceListStatus);

    if (TF_GetCode(sessionStatus) != TF_OK) {
        RAI_SetError(error, RAI_EMODELCREATE, RedisModule_Strdup(TF_Message(status)));
        goto cleanup;
    }

    TF_DeleteSessionOptions(sessionOptions);
    TF_DeleteStatus(sessionStatus);

    model->model = graph;
    model->session = session;
    return REDISMODULE_OK;

cleanup:
    TF_DeleteGraph(graph);
    if (options)
        TF_DeleteImportGraphDefOptions(options);
    if (tfbuffer)
        TF_DeleteBuffer(tfbuffer);
    if (status)
        TF_DeleteStatus(status);
    if (sessionOptions)
        TF_DeleteSessionOptions(sessionOptions);
    if (sessionStatus)
        TF_DeleteStatus(sessionStatus);
    return REDISMODULE_ERR;
}

void RAI_ModelFreeTF(RAI_Model *model, RAI_Error *error) {

    // If we got an error before we created the session, there's nothing to free.
    if (model->session == NULL) {
        return;
    }

    TF_Status *status = TF_NewStatus();
    TF_CloseSession(model->session, status);
    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELFREE, TF_Message(status));
        return;
    }

    TF_DeleteSession(model->session, status);
    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELFREE, TF_Message(status));
        return;
    }

    TF_DeleteGraph(model->model);
    TF_DeleteStatus(status);
}

int RAI_ModelRunTF(RAI_ModelRunCtx **mctxs, RAI_Error *error) {
    TF_Status *status = TF_NewStatus();

    const size_t nbatches = array_len(mctxs);
    if (nbatches == 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR No batches to run");
        return 1;
    }

    const size_t ninputs = array_len(mctxs[0]->inputs);
    const size_t noutputs = array_len(mctxs[0]->outputs);
    TF_Tensor *inputTensorsValues[ninputs];
    TF_Output inputs[ninputs];
    TF_Tensor *outputTensorsValues[noutputs];
    TF_Output outputs[noutputs];

    size_t batch_sizes[nbatches];
    size_t batch_offsets[nbatches];
    size_t total_batch_size = 0;
    if (ninputs > 0) {
        for (size_t b = 0; b < nbatches; ++b) {
            batch_sizes[b] = RAI_TensorDim(mctxs[b]->inputs[0].tensor, 0);
            total_batch_size += batch_sizes[b];
        }
        batch_offsets[0] = 0;
        for (size_t b = 1; b < nbatches; ++b) {
            batch_offsets[b] = batch_offsets[b - 1] + batch_sizes[b - 1];
        }
    }

    for (size_t i = 0; i < ninputs; ++i) {
        RAI_Tensor *batched_input_tensors[nbatches];

        for (size_t b = 0; b < nbatches; ++b) {
            batched_input_tensors[b] = mctxs[b]->inputs[i].tensor;
        }
        inputTensorsValues[i] = RAI_TFTensorFromTensors(batched_input_tensors, nbatches);
        TF_Output port;
        port.oper = TF_GraphOperationByName(mctxs[0]->model->model, mctxs[0]->inputs[i].name);
        port.index = 0;
        if (port.oper == NULL) {
            return 1;
        }
        inputs[i] = port;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        TF_Output port;
        port.oper = TF_GraphOperationByName(mctxs[0]->model->model, mctxs[0]->outputs[i].name);
        port.index = 0;
        if (port.oper == NULL) {
            return 1;
        }
        outputs[i] = port;
    }

    TF_SessionRun(mctxs[0]->model->session, NULL /* run_options */, inputs, inputTensorsValues,
                  ninputs, outputs, outputTensorsValues, noutputs, NULL /* target_opers */,
                  0 /* ntargets */, NULL /* run_Metadata */, status);

    for (size_t i = 0; i < ninputs; ++i) {
        TF_DeleteTensor(inputTensorsValues[i]);
    }

    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELRUN, errorMessage);
        TF_DeleteStatus(status);
        RedisModule_Free(errorMessage);
        return 1;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        if (nbatches > 1) {
            if (TF_NumDims(outputTensorsValues[i]) == 0) {
                continue;
            }
            if (TF_Dim(outputTensorsValues[i], 0) != total_batch_size) {
                TF_DeleteTensor(outputTensorsValues[i]);
                TF_DeleteStatus(status);
                RAI_SetError(error, RAI_EMODELRUN,
                             "ERR Model did not generate the expected batch size");
                return 1;
            }

            for (size_t b = 0; b < nbatches; b++) {
                mctxs[b]->outputs[i].tensor = RAI_TensorCreateFromTFTensor(
                    outputTensorsValues[i], batch_offsets[b], batch_sizes[b]);
            }
        } else {
            mctxs[0]->outputs[i].tensor =
                RAI_TensorCreateFromTFTensor(outputTensorsValues[i], 0, -1);
        }
        TF_DeleteTensor(outputTensorsValues[i]);
    }

    TF_DeleteStatus(status);

    return 0;
}

int RAI_ModelSerializeTF(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {

    if (model->data) {
        *buffer = RedisModule_Calloc(model->datalen, sizeof(char));
        memcpy(*buffer, model->data, model->datalen);
        *len = model->datalen;
    } else {
        TF_Buffer *tf_buffer = TF_NewBuffer();
        TF_Status *status = TF_NewStatus();

        TF_GraphToGraphDef(model->model, tf_buffer, status);

        if (TF_GetCode(status) != TF_OK) {
            RAI_SetError(error, RAI_EMODELSERIALIZE, "ERR Error serializing TF model");
            TF_DeleteBuffer(tf_buffer);
            TF_DeleteStatus(status);
            return 1;
        }

        *buffer = RedisModule_Alloc(tf_buffer->length);
        memcpy(*buffer, tf_buffer->data, tf_buffer->length);
        *len = tf_buffer->length;

        TF_DeleteBuffer(tf_buffer);
        TF_DeleteStatus(status);
    }

    return 0;
}

const char *RAI_GetBackendVersionTF(void) { return TF_Version(); }
