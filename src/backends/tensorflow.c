#define REDISMODULE_MAIN
#include "backends/util.h"
#include "backends/tensorflow.h"
#include "util/arr.h"
#include "execution/execution_contexts/modelRun_ctx.h"
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
        case 64:
            return TF_DOUBLE;
        default:
            return 0;
        }
    } else if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        case 8:
            return TF_INT8;
        case 16:
            return TF_INT16;
        case 32:
            return TF_INT32;
        case 64:
            return TF_INT64;
        default:
            return 0;
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            return TF_UINT8;
        case 16:
            return TF_UINT16;
        default:
            return 0;
        }
    } else if (dtype.code == kDLBool) {
        switch (dtype.bits) {
        case 8:
            return TF_BOOL;
        default:
            return 0;
        }
    } else if (dtype.code == kDLString) {
        switch (dtype.bits) {
        case 8:
            return TF_STRING;
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
    case TF_BOOL:
        return (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1};
    case TF_STRING:
        return (DLDataType){.code = kDLString, .bits = 8, .lanes = 1};
    default:
        return (DLDataType){.bits = 0};
    }
}

RAI_Tensor *RAI_TensorCreateFromTFTensor(TF_Tensor *tensor, size_t batch_offset,
                                         long long batch_size) {

    int n_dims = TF_NumDims(tensor);
    int64_t total_batch_size = TF_Dim(tensor, 0);
    total_batch_size = total_batch_size > 0 ? total_batch_size : 1;

    size_t shape[n_dims];
    for (int i = 0; i < n_dims; ++i) {
        shape[i] = TF_Dim(tensor, i);
    }
    if (batch_size != -1) {
        shape[0] = batch_size; // the TF tensor was batched
    } else {
        batch_size = total_batch_size; // the TF tensor wasn't batched
    }

    DLDataType data_type = RAI_GetDLDataTypeFromTF(TF_TensorType(tensor));
    RAI_Tensor *out = RAI_TensorNew(data_type, shape, n_dims);
    size_t out_tensor_len = RAI_TensorLength(out);

    if (data_type.code == kDLString) {
        TF_TString *tensor_data = TF_TensorData(tensor);
        const char *strings_data[out_tensor_len];
        size_t strings_lengths[out_tensor_len];

        // Calculate the blob size for this tensor and allocate space for it
        size_t element_index = batch_offset * (out_tensor_len / batch_size);
        size_t blob_len = 0;
        uint64_t *offsets = RAI_TensorStringElementsOffsets(out);
        offsets[0] = 0;
        for (size_t i = 0; i < out_tensor_len; i++) {
            size_t str_element_len = TF_TString_GetSize(tensor_data + element_index);
            strings_lengths[i] = str_element_len;
            strings_data[i] = TF_TString_GetDataPointer(tensor_data + element_index++);
            offsets[i] = blob_len;
            blob_len += str_element_len;
            if (strings_data[i][str_element_len - 1] != '\0') {
                blob_len++; // Add space for null character at the end of every string
            }
        }
        out->blobSize = blob_len;
        out->tensor.dl_tensor.data = RedisModule_Calloc(1, blob_len);

        // Go over again and set tensor data elements one by one
        element_index = batch_offset * (out_tensor_len / batch_size);
        char *tensor_blob = RAI_TensorData(out);
        for (size_t i = 0; i < out_tensor_len; i++) {
            memcpy(tensor_blob + offsets[i], strings_data[i], strings_lengths[i]);
            TF_TString_Dealloc(tensor_data + element_index++);
        }
    } else {
        size_t non_batched_tensor_size = TF_TensorByteSize(tensor) / total_batch_size;
        size_t blob_len = non_batched_tensor_size * batch_size;
        out->tensor.dl_tensor.data = RedisModule_Alloc(blob_len);
        out->blobSize = blob_len;
        memcpy(RAI_TensorData(out), TF_TensorData(tensor) + non_batched_tensor_size * batch_offset,
               blob_len);
    }
    return out;
}

void RAI_TFDeallocator(void *data, size_t len, void *arg) {
    // printf("DEALLOCATOR CALLED\n");
    // do nothing, memory is managed by Redis
}

TF_Tensor *RAI_TFTensorFromTensors(RAI_Tensor **tensors, size_t count) {
    RedisModule_Assert(count > 0);

    int64_t batch_size = 0;
    size_t batch_byte_size = 0;

    for (size_t i = 0; i < count; i++) {
        batch_size += RAI_TensorDim(tensors[i], 0);
        batch_byte_size += RAI_TensorByteSize(tensors[i]);
    }

    // get the shapes of the batched tensor: all inner dims should be the same,
    // so we go over t0 dims.
    RAI_Tensor *t0 = tensors[0];
    int n_dim = RAI_TensorNumDims(t0);
    int64_t batched_shape[n_dim];
    batched_shape[0] = batch_size;
    size_t batched_tensor_len = batch_size;
    for (int i = 1; i < n_dim; i++) {
        batched_shape[i] = RAI_TensorDim(t0, i);
        batched_tensor_len *= batched_shape[i];
    }

    TF_Tensor *out = NULL;
    if (RAI_TensorDataType(t0).code == kDLString) {
        out = TF_AllocateTensor(TF_STRING, batched_shape, RAI_TensorNumDims(t0),
                                sizeof(TF_TString) * batched_tensor_len);
        // go over the string elements and copy the data to the TF tensor
        TF_TString *tf_str = (TF_TString *)TF_TensorData(out);
        size_t element_ind = 0;
        for (size_t i = 0; i < count; i++) {
            RAI_Tensor *t = tensors[i];
            uint64_t *offsets = RAI_TensorStringElementsOffsets(t);
            for (size_t j = 0; j < RAI_TensorLength(t); j++) {
                TF_TString_Init(&tf_str[element_ind]);
                size_t str_len = j < RAI_TensorLength(t) - 1 ? offsets[j + 1] - offsets[j]
                                                             : RAI_TensorByteSize(t) - offsets[j];
                TF_TString_Copy(&tf_str[element_ind++], RAI_TensorData(t) + offsets[j], str_len);
            }
        }
    } else if (count > 1) {
        out = TF_AllocateTensor(RAI_GetTFDataTypeFromDL(RAI_TensorDataType(t0)), batched_shape,
                                RAI_TensorNumDims(t0), batch_byte_size);
        size_t offset = 0;
        for (size_t i = 0; i < count; i++) {
            size_t tensor_byte_size = RAI_TensorByteSize(tensors[i]);
            memcpy(TF_TensorData(out) + offset, RAI_TensorData(tensors[i]), tensor_byte_size);
            offset += tensor_byte_size;
        }
    } else {
        out = TF_NewTensor(RAI_GetTFDataTypeFromDL(RAI_TensorDataType(t0)), RAI_TensorShape(t0),
                           RAI_TensorNumDims(t0), RAI_TensorData(t0), RAI_TensorByteSize(t0),
                           &RAI_TFDeallocator, NULL);
    }
    return out;
}

RAI_Model *RAI_ModelCreateTF(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                             size_t ninputs, const char **inputs, size_t noutputs,
                             const char **outputs, const char *modeldef, size_t modellen,
                             RAI_Error *error) {
    RAI_Device device;
    int64_t deviceid;

    if (!parseDeviceStr(devicestr, &device, &deviceid)) {
        RAI_SetError(error, RAI_EMODELIMPORT, "ERR unsupported device");
    }

    TF_Graph *model = TF_NewGraph();
    TF_Status *status = TF_NewStatus();
    TF_Buffer *tfbuffer = TF_NewBuffer();
    TF_ImportGraphDefOptions *options = TF_NewImportGraphDefOptions();
    TF_Status *optionsStatus = NULL;
    TF_SessionOptions *sessionOptions = NULL;
    TF_Status *sessionStatus = NULL;
    TF_Session *session = NULL;

    tfbuffer->length = modellen;
    tfbuffer->data = modeldef;

    TF_GraphImportGraphDef(model, tfbuffer, options, status);

    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELIMPORT, errorMessage);
        RedisModule_Free(errorMessage);
        return NULL;
    }

    for (size_t i = 0; i < ninputs; ++i) {
        TF_Operation *oper = TF_GraphOperationByName(model, inputs[i]);
        if (oper == NULL || strcmp(TF_OperationOpType(oper), "Placeholder") != 0) {
            size_t len = strlen(inputs[i]);
            char *msg = RedisModule_Calloc(60 + len, sizeof(*msg));
            sprintf(msg, "ERR Input node named \"%s\" not found in TF graph.", inputs[i]);
            RAI_SetError(error, RAI_EMODELIMPORT, msg);
            RedisModule_Free(msg);
            goto cleanup;
        }
    }

    for (size_t i = 0; i < noutputs; ++i) {
        TF_Operation *oper = TF_GraphOperationByName(model, outputs[i]);
        if (oper == NULL) {
            size_t len = strlen(outputs[i]);
            char *msg = RedisModule_Calloc(60 + len, sizeof(*msg));
            sprintf(msg, "ERR Output node named \"%s\" not found in TF graph", outputs[i]);
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
    session = TF_NewSession(model, sessionOptions, sessionStatus);

    TF_Status *deviceListStatus = TF_NewStatus();
    TF_DeviceList *deviceList = TF_SessionListDevices(session, deviceListStatus);
    const int num_devices = TF_DeviceListCount(deviceList);
    int foundNoGPU = 1;
    for (int i = 0; i < num_devices; ++i) {
        const char *device_type = TF_DeviceListType(deviceList, i, deviceListStatus);
        int cmp = strcmp(device_type, "GPU");
        if (cmp == 0) {
            foundNoGPU = 0;
            break;
        }
    }
    if (foundNoGPU == 1 && device == RAI_DEVICE_GPU) {
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

    char **inputs_ = array_new(char *, ninputs);
    for (long long i = 0; i < ninputs; i++) {
        inputs_ = array_append(inputs_, RedisModule_Strdup(inputs[i]));
    }

    char **outputs_ = array_new(char *, noutputs);
    for (long long i = 0; i < noutputs; i++) {
        outputs_ = array_append(outputs_, RedisModule_Strdup(outputs[i]));
    }

    char *buffer = RedisModule_Calloc(modellen, sizeof(*buffer));
    memcpy(buffer, modeldef, modellen);

    RAI_Model *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->model = model;
    ret->session = session;
    ret->backend = backend;
    ret->devicestr = RedisModule_Strdup(devicestr);
    ret->ninputs = ninputs;
    ret->inputs = inputs_;
    ret->noutputs = noutputs;
    ret->outputs = outputs_;
    ret->opts = opts;
    ret->refCount = 1;
    ret->data = buffer;
    ret->datalen = modellen;

    return ret;

cleanup:
    TF_DeleteGraph(model);
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
    return NULL;
}

void RAI_ModelFreeTF(RAI_Model *model, RAI_Error *error) {
    TF_Status *status = TF_NewStatus();
    TF_CloseSession(model->session, status);

    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELFREE, RedisModule_Strdup(TF_Message(status)));
        return;
    }

    TF_DeleteSession(model->session, status);
    model->session = NULL;

    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELFREE, RedisModule_Strdup(TF_Message(status)));
        return;
    }

    TF_DeleteGraph(model->model);
    model->model = NULL;

    RedisModule_Free(model->devicestr);

    if (model->inputs) {
        size_t ninputs = array_len(model->inputs);
        for (size_t i = 0; i < ninputs; i++) {
            RedisModule_Free(model->inputs[i]);
        }
        array_free(model->inputs);
    }

    if (model->outputs) {
        size_t noutputs = array_len(model->outputs);
        for (size_t i = 0; i < noutputs; i++) {
            RedisModule_Free(model->outputs[i]);
        }
        array_free(model->outputs);
    }

    if (model->data) {
        RedisModule_Free(model->data);
    }

    TF_DeleteStatus(status);
}

int RAI_ModelRunTF(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error) {
    int res = REDISMODULE_ERR;
    const size_t nbatches = array_len(ectxs);
    if (nbatches == 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR No batches to run");
        return res;
    }

    TF_Status *status = TF_NewStatus();
    size_t ninputs = RAI_ExecutionCtx_NumInputs(ectxs[0]);
    size_t noutputs = RAI_ExecutionCtx_NumOutputs(ectxs[0]);

    TF_Input inputs[ninputs];
    TF_Output outputs[noutputs];

    TF_Tensor *inputTensorsValues[ninputs];
    TF_Tensor *outputTensorsValues[noutputs];

    size_t batch_sizes[nbatches];
    size_t batch_offsets[nbatches];
    size_t total_batch_size = 0;
    if (ninputs > 0) {
        for (size_t b = 0; b < nbatches; ++b) {
            batch_sizes[b] = RAI_TensorDim(RAI_ExecutionCtx_GetInput(ectxs[b], 0), 0);
            total_batch_size += batch_sizes[b];
        }
        batch_offsets[0] = 0;
        for (size_t b = 1; b < nbatches; ++b) {
            batch_offsets[b] = batch_offsets[b - 1] + batch_sizes[b - 1];
        }
    }

    void *tfGraph = RAI_ModelGetModel(model);
    void *tfSession = RAI_ModelGetSession(model);

    for (size_t i = 0; i < ninputs; ++i) {
        RAI_Tensor *batched_input_tensors[nbatches];
        for (size_t b = 0; b < nbatches; ++b) {
            batched_input_tensors[b] = RAI_ExecutionCtx_GetInput(ectxs[b], i);
        }
        inputTensorsValues[i] = RAI_TFTensorFromTensors(batched_input_tensors, nbatches);
        TF_Input port;
        port.oper = TF_GraphOperationByName(tfGraph, RAI_ModelGetInputName(model, i));
        // this operation must exist in the graph (verified when creating the model)
        RedisModule_Assert(port.oper);
        port.index = 0;
        inputs[i] = port;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        TF_Output port;
        port.oper = TF_GraphOperationByName(tfGraph, RAI_ModelGetOutputName(model, i));
        // this operation must exist in the graph (verified when creating the model)
        RedisModule_Assert(port.oper);
        port.index = 0;
        outputs[i] = port;
    }

    TF_SessionRun(tfSession, NULL /* run_options */, inputs, inputTensorsValues, ninputs, outputs,
                  outputTensorsValues, noutputs, NULL /* target_opers */, 0 /* ntargets */,
                  NULL /* run_Metadata */, status);

    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELRUN, TF_Message(status));
        goto cleanup;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        if (nbatches > 1) {
            if (TF_NumDims(outputTensorsValues[i]) == 0) {
                continue;
            }
            if (TF_Dim(outputTensorsValues[i], 0) != total_batch_size) {
                TF_DeleteTensor(outputTensorsValues[i]);
                RAI_SetError(error, RAI_EMODELRUN,
                             "ERR Tensorflow batching error: model did not generate the expected "
                             "batch size");
                goto cleanup;
            }

            for (size_t b = 0; b < nbatches; b++) {
                RAI_ExecutionCtx_SetOutput(ectxs[b],
                                           RAI_TensorCreateFromTFTensor(outputTensorsValues[i],
                                                                        batch_offsets[b],
                                                                        batch_sizes[b]),
                                           i);
            }
        } else {
            RAI_ExecutionCtx_SetOutput(
                ectxs[0], RAI_TensorCreateFromTFTensor(outputTensorsValues[i], 0, -1), i);
        }
    }
    res = REDISMODULE_OK;

cleanup:
    TF_DeleteStatus(status);
    for (size_t i = 0; i < ninputs; ++i) {
        RAI_Tensor *t = RAI_ExecutionCtx_GetInput(ectxs[0], i);
        // free the underline string buffer for every element
        if (RAI_TensorDataType(t).code == kDLString) {
            TF_TString *tf_strings = TF_TensorData(inputTensorsValues[i]);
            for (size_t j = 0; j < RAI_TensorLength(t); j++) {
                TF_TString_Dealloc(tf_strings + j);
            }
        }
        TF_DeleteTensor(inputTensorsValues[i]);
    }
    for (size_t i = 0; i < noutputs; i++) {
        TF_DeleteTensor(outputTensorsValues[i]);
    }
    return res;
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
