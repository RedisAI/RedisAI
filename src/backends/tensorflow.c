#define REDISMODULE_MAIN
#include "backends/tensorflow.h"
#include "backends/util.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "model.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"

#define RAI_TF_FN_NAME "rai_tf_forward"

TF_CAPI_EXPORT extern void *TFE_HandleToDLPack(TFE_TensorHandle *h, TF_Status *status);

TF_CAPI_EXPORT extern TFE_TensorHandle *TFE_HandleFromDLPack(void *dlm, TF_Status *status,
                                                             TFE_Context *ctx);

TF_CAPI_EXPORT extern void TFE_CallDLManagedTensorDeleter(void *dlm_ptr);

int RAI_InitBackendTF(int (*get_api_fn)(const char *, void *)) {
    get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
    get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
    get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
    get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
    get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));

    return REDISMODULE_OK;
}

struct TFDLManagedTensorCtx {
    TFE_TensorHandle *reference;
    int64_t ndim;
    int64_t *shape;
    int64_t *strides;
    DLManagedTensor tensor;
};
typedef struct TFDLManagedTensorCtx TFDLManagedTensorCtx;

TFDLManagedTensorCtx *TFDLManagedTensorCtx_Create(TFE_TensorHandle *h, TF_Status *status) {
    TFDLManagedTensorCtx *ctx = RedisModule_Alloc(sizeof(TFDLManagedTensorCtx));
    ctx->reference = h;
    ctx->ndim = TFE_TensorHandleNumDims(h, status);
    ctx->shape = RedisModule_Calloc(ctx->ndim, sizeof(int64_t));
    ctx->strides = RedisModule_Calloc(ctx->ndim, sizeof(int64_t));
    for (int i = 0; i < ctx->ndim; i++) {
        ctx->shape[i] = TFE_TensorHandleDim(h, i, status);
        ctx->strides[i] = 1;
    }
    for (int i = ctx->ndim - 2; i >= 0; i--) {
        ctx->strides[i] = ctx->shape[i + 1] * ctx->strides[i + 1];
    }
    return ctx;
}

void TFDLManagedTensorCtx_Free(TFDLManagedTensorCtx *ctx) {
    RedisModule_Free(ctx->shape);
    RedisModule_Free(ctx->strides);
    RedisModule_Free(ctx);
}

void DLManagedTensorDeleter(DLManagedTensor *arg) {
    TFDLManagedTensorCtx *owner = (TFDLManagedTensorCtx *)(arg->manager_ctx);
    TFE_DeleteTensorHandle(owner->reference);
    TFDLManagedTensorCtx_Free(owner);
}

DLDataType GetDLDataType(TF_DataType data_type, TF_Status *status) {
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = TF_DataTypeSize(data_type) * 8;
    switch (data_type) {
    case TF_HALF:
    case TF_FLOAT:
    case TF_DOUBLE:
        dtype.code = kDLFloat;
        break;
    case TF_INT8:
    case TF_INT16:
    case TF_INT32:
    case TF_INT64:
        dtype.code = kDLInt;
        break;
    case TF_BOOL:
    case TF_UINT8:
    case TF_UINT16:
    case TF_UINT32:
    case TF_UINT64:
        dtype.code = kDLUInt;
        break;
    case TF_BFLOAT16:
        dtype.code = kDLBfloat;
        break;
    default:
        //     err data_type  " is not supported by dlpack");
        break;
    }
    return dtype;
}

DLDevice GetDLDevice(TFE_TensorHandle *h, TF_Status *status) {
    DLDevice device;
    const char *device_name = TFE_TensorHandleBackingDeviceName(h, status);

    char device_type[64];
    int device_id = 0;
    if (strncasecmp(device_name, "/device:", 8) == 0) {
        strncpy(device_type, device_name + 8, 3);
        if (strlen(device_name) > 8 + 4) {
            device_id = atoi(device_name + 8 + 4);
        }
    } else {
        strncpy(device_type, device_name, 3);
        if (strlen(device_name) > 4) {
            device_id = atoi(device_name + 4);
        }
    }

    device.device_id = device_id;
    if (strcasecmp(device_type, "CPU") == 0) {
        device.device_type = kDLCPU;
    } else if (strcasecmp(device_type, "GPU") == 0) {
        device.device_type = kDLGPU;
    } else {
        // TODO err "Unsupported Device Type for dlpack"
    }

    return device;
}

int DeviceNameFromDLContext(const DLDevice *device, char device_name[64]) {
    switch (device->device_type) {
    case kDLCPU:
        strcpy(device_name, "CPU:0");
        return 0;
    case kDLGPU:
        sprintf(device_name, "GPU:%d", device->device_id);
        return 0;
    }
    return 1;
}

int TFDataTypeFromDLDataType(const DLDataType *dtype, TF_DataType *tf_dtype) {
    switch (dtype->code) {
    case kDLUInt:
        switch (dtype->bits) {
        case 8:
            *tf_dtype = TF_UINT8;
            return 0;
        case 16:
            *tf_dtype = TF_UINT16;
            return 0;
        case 32:
            *tf_dtype = TF_UINT32;
            return 0;
        case 64:
            *tf_dtype = TF_UINT64;
            return 0;
        default:
            return 1;
        }
        return 0;
    case kDLInt:
        switch (dtype->bits) {
        case 8:
            *tf_dtype = TF_INT8;
            return 0;
        case 16:
            *tf_dtype = TF_INT16;
            return 0;
        case 32:
            *tf_dtype = TF_INT32;
            return 0;
        case 64:
            *tf_dtype = TF_INT64;
            return 0;
        default:
            return 1;
        }
        return 1;
    case kDLFloat:
        switch (dtype->bits) {
        case 16:
            *tf_dtype = TF_HALF;
            return 0;
        case 32:
            *tf_dtype = TF_FLOAT;
            return 0;
        case 64:
            *tf_dtype = TF_DOUBLE;
            return 0;
        default:
            return 1;
        }
        break;
    case kDLBfloat:
        switch (dtype->bits) {
        case 16:
            *tf_dtype = TF_BFLOAT16;
            return 0;
        default:
            return 1;
        }
        break;
    default:
        return 1;
    }
}

void DeallocatorWrapperFunc(void *data, size_t len, void *dlmt_vptr) {
    // NOTE: in the original TF implementation, the TFE_NewTensorHandleFromDeviceMemory
    // function takes ownership of the device memory. The following function call is
    // performed in order to deallocate the underlying DLPack structure
    // In our case we are making the call from TFE_HandleFromDLPack, so the memory
    // is already managed by the DLPack managed tensor that originally created it,
    // and it is regulated by reference counting from within RedisAI.
    // Therefore the present function should do nothing, the comment is retained
    // for clarity only.
    // TFE_CallDLManagedTensorDeleter(dlmt_vptr);
}

bool IsValidStrideCompactRowMajorData(int64_t *shape_arr, int64_t *stride_arr, int ndim) {
    if (ndim >= 1 && stride_arr[ndim - 1] != 1) {
        return false;
    }
    for (int i = ndim - 2; i >= 0; --i) {
        if (stride_arr[i] != shape_arr[i + 1] * stride_arr[i + 1]) {
            return false;
        }
    }
    return true;
}

void TFE_CallDLManagedTensorDeleter(void *dlm_ptr) {
    DLManagedTensor *dlMTensor = (DLManagedTensor *)dlm_ptr;
    if (dlMTensor->deleter != NULL) {
        dlMTensor->deleter(dlMTensor);
    }
}

void *TFE_HandleToDLPack(TFE_TensorHandle *h, TF_Status *status) {
    DLDevice tf_dlm_device = GetDLDevice(h, status);
    if (TF_GetCode(status) != TF_OK) {
        return NULL;
    }

    void *tf_dlm_data = TFE_TensorHandleDevicePointer(h, status);
    if (TF_GetCode(status) != TF_OK) {
        return NULL;
    }

    TF_DataType data_type = TFE_TensorHandleDataType(h);

    DLDataType tf_dlm_type = GetDLDataType(data_type, status);
    if (TF_GetCode(status) != TF_OK) {
        return NULL;
    }

    TFDLManagedTensorCtx *tf_dlm_tensor_ctx = TFDLManagedTensorCtx_Create(h, status);

    DLManagedTensor *dlm_tensor = &tf_dlm_tensor_ctx->tensor;
    dlm_tensor->manager_ctx = tf_dlm_tensor_ctx;
    dlm_tensor->deleter = &DLManagedTensorDeleter;
    dlm_tensor->dl_tensor.device = tf_dlm_device;
    dlm_tensor->dl_tensor.ndim = tf_dlm_tensor_ctx->ndim;
    dlm_tensor->dl_tensor.data = tf_dlm_data;
    dlm_tensor->dl_tensor.dtype = tf_dlm_type;
    dlm_tensor->dl_tensor.shape = tf_dlm_tensor_ctx->shape;
    dlm_tensor->dl_tensor.strides = tf_dlm_tensor_ctx->strides;
    dlm_tensor->dl_tensor.byte_offset = 0;

    return (void *)dlm_tensor;
}

TFE_TensorHandle *TFE_HandleFromDLPack(void *dlm, TF_Status *status, TFE_Context *ctx) {
    DLManagedTensor *dlmt = (DLManagedTensor *)dlm;
    DLTensor *dl_tensor = &dlmt->dl_tensor;
    char device_name[64];
    int ret = DeviceNameFromDLContext(&dl_tensor->device, device_name);
    if (ret != 0) {
        // TODO Unsupported device type
        return NULL;
    }
    TF_DataType dtype;
    ret = TFDataTypeFromDLDataType(&dl_tensor->dtype, &dtype);
    if (ret != 0) {
        // TODO Unsupported data type
        return NULL;
    }
    int num_dims = dl_tensor->ndim;
    const int64_t *dims = dl_tensor->shape;
    void *data = dl_tensor->data;

    size_t total_bytes = dl_tensor->dtype.bits / 8;
    for (int i = 0; i < num_dims; i++) {
        total_bytes *= dims[i];
    }

    if (dl_tensor->strides != NULL &&
        !IsValidStrideCompactRowMajorData(dl_tensor->shape, dl_tensor->strides, num_dims)) {
        // TODO err "Invalid strides array from DLPack"
        return NULL;
    }

    TFE_TensorHandle *handle =
        TFE_NewTensorHandleFromDeviceMemory(ctx, device_name, dtype, dims, num_dims, data,
                                            total_bytes, &DeallocatorWrapperFunc, dlmt, status);

    return handle;
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

    TF_Graph *graph = TF_NewGraph();
    TF_ImportGraphDefOptions *options = TF_NewImportGraphDefOptions();
    TF_Status *status = TF_NewStatus();
    TF_Buffer *tfbuffer = TF_NewBuffer();

    tfbuffer->length = modellen;
    tfbuffer->data = modeldef;

    TF_GraphImportGraphDef(graph, tfbuffer, options, status);

    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELIMPORT, errorMessage);
        RedisModule_Free(errorMessage);
        return NULL;
    }

    for (size_t i = 0; i < ninputs; ++i) {
        TF_Operation *oper = TF_GraphOperationByName(graph, inputs[i]);
        if (oper == NULL || strcmp(TF_OperationOpType(oper), "Placeholder") != 0) {
            size_t len = strlen(inputs[i]);
            char *msg = RedisModule_Calloc(60 + len, sizeof(*msg));
            sprintf(msg, "ERR Input node named \"%s\" not found in TF graph.", inputs[i]);
            RAI_SetError(error, RAI_EMODELIMPORT, msg);
            RedisModule_Free(msg);
            return NULL;
        }
    }

    for (size_t i = 0; i < noutputs; ++i) {
        TF_Operation *oper = TF_GraphOperationByName(graph, outputs[i]);
        if (oper == NULL) {
            size_t len = strlen(outputs[i]);
            char *msg = RedisModule_Calloc(60 + len, sizeof(*msg));
            sprintf(msg, "ERR Output node named \"%s\" not found in TF graph", outputs[i]);
            RAI_SetError(error, RAI_EMODELIMPORT, msg);
            RedisModule_Free(msg);
            return NULL;
        }
    }

    TF_DeleteImportGraphDefOptions(options);
    options = NULL;
    TF_DeleteBuffer(tfbuffer);
    tfbuffer = NULL;

    TF_Output tf_inputs[ninputs];
    TF_Output tf_outputs[noutputs];

    for (size_t i = 0; i < ninputs; ++i) {
        TF_Output port;
        port.oper = TF_GraphOperationByName(graph, inputs[i]);
        port.index = 0;
        if (port.oper == NULL) {
            return NULL;
        }
        tf_inputs[i] = port;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        TF_Output port;
        port.oper = TF_GraphOperationByName(graph, outputs[i]);
        port.index = 0;
        if (port.oper == NULL) {
            return NULL;
        }
        tf_outputs[i] = port;
    }

    TF_Function *function =
        TF_GraphToFunction(graph,                // fn_body
                           RAI_TF_FN_NAME, 0,    // fn_name, append_hash_to_fn_name,
                           -1, NULL,             // num_opers, opers
                           ninputs, tf_inputs,   // ninputs, inputs,
                           noutputs, tf_outputs, // noutputs, outputs
                           outputs,              // output_names,
                           NULL,                 // opts
                           NULL,                 // description
                           status                // status
        );

    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(status)));
        goto cleanup;
    }

    // For setting config options in session from the C API see:
    // https://github.com/tensorflow/tensorflow/issues/13853
    // import tensorflow as tf
    // config = tf.ConfigProto(device_count = {'GPU': 0})
    // serialized = config.SerializeToString()
    // result = list(map(hex, serialized))
    // print(result)

    TFE_ContextOptions *context_opts = TFE_NewContextOptions();

    if (device == RAI_DEVICE_CPU) {
        // Set number of GPU to 0 with
        // config.device_count = {'GPU': 0}
        uint8_t config[] = {0x0a, 0x07, 0x0a, 0x03, 0x47, 0x50, 0x55, 0x10, 0x00};
        TFE_ContextOptionsSetConfig(context_opts, (void *)config, sizeof(config), status);

        if (TF_GetCode(status) != TF_OK) {
            RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(status)));
            goto cleanup;
        }

        if (opts.backends_intra_op_parallelism > 0) {
            uint8_t proto[] = {0x10, (uint8_t)opts.backends_intra_op_parallelism};
            TFE_ContextOptionsSetConfig(context_opts, proto, sizeof(proto), status);
            if (TF_GetCode(status) != TF_OK) {
                RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(status)));
                goto cleanup;
            }
        }

        if (opts.backends_inter_op_parallelism > 0) {
            uint8_t proto1[] = {0x28, (uint8_t)opts.backends_inter_op_parallelism};
            TFE_ContextOptionsSetConfig(context_opts, proto1, sizeof(proto1), status);
            if (TF_GetCode(status) != TF_OK) {
                RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(status)));
                goto cleanup;
            }
        }
    } else if (device == RAI_DEVICE_GPU) {
        if (deviceid == -1) {
            // Set
            // config.gpu_options.allow_growth = True
            uint8_t config[4] = {0x32, 0x02, 0x20, 0x01};
            TFE_ContextOptionsSetConfig(context_opts, (void *)config, 4, status);
        } else {
            // Set
            // config.gpu_options.allow_growth = True
            // config.gpu_options.visible_device_list = '<deviceid>'
            uint8_t config[7] = {0x32, 0x05, 0x20, 0x01, 0x2a, 0x01, 0x30};
            config[6] += (uint8_t)deviceid;
            TFE_ContextOptionsSetConfig(context_opts, (void *)config, 7, status);
        }
    }

    TFE_ContextOptionsSetAsync(context_opts, 0);
    TFE_ContextOptionsSetDevicePlacementPolicy(context_opts, TFE_DEVICE_PLACEMENT_EXPLICIT);

    TFE_Context *context = TFE_NewContext(context_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(status)));
        goto cleanup;
    }

    TFE_ContextAddFunction(context, function, status);
    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, RedisModule_Strdup(TF_Message(status)));
        goto cleanup;
    }

    TFE_DeleteContextOptions(context_opts);

    TF_DeviceList *deviceList = TFE_ContextListDevices(context, status);
    const int num_devices = TF_DeviceListCount(deviceList);
    int foundNoGPU = 1;
    for (int i = 0; i < num_devices; ++i) {
        const char *device_type = TF_DeviceListType(deviceList, i, status);
        int cmp = strcmp(device_type, "GPU");
        if (cmp == 0) {
            foundNoGPU = 0;
            break;
        }
    }
    if (foundNoGPU == 1 && device == RAI_DEVICE_GPU) {
        RAI_SetError(error, RAI_EMODELCREATE, "ERR GPU requested but TF couldn't find CUDA");
        TF_DeleteDeviceList(deviceList);
        TF_DeleteStatus(status);
        goto cleanup;
    }
    TF_DeleteDeviceList(deviceList);

    if (TF_GetCode(status) != TF_OK) {
        RAI_SetError(error, RAI_EMODELCREATE, RedisModule_Strdup(TF_Message(status)));
        goto cleanup;
    }

    TF_DeleteStatus(status);

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
    ret->model = graph;
    ret->session = context;
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
    TF_DeleteGraph(graph);
    if (options)
        TF_DeleteImportGraphDefOptions(options);
    if (tfbuffer)
        TF_DeleteBuffer(tfbuffer);
    if (status)
        TF_DeleteStatus(status);
    return NULL;
}

void RAI_ModelFreeTF(RAI_Model *model, RAI_Error *error) {
    TFE_DeleteContext(model->session);
    model->session = NULL;

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
    TFE_TensorHandle *inputTensorsHandles[ninputs];
    TFE_TensorHandle *outputTensorsHandles[noutputs];
    TFE_TensorHandle *deviceInputTensorsHandles[ninputs];
    TFE_TensorHandle *deviceOutputTensorsHandles[noutputs];

    bool on_cpu = false;
    if (strncasecmp(mctxs[0]->model->devicestr, "CPU", 3) == 0) {
        on_cpu == true;
    }

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

    char tf_devicestr[256];
    int devicestr_len = strlen(mctxs[0]->model->devicestr);
    if (on_cpu) {
        sprintf(tf_devicestr, "/device:CPU:0");
    } else if (devicestr_len == 3) {
        sprintf(tf_devicestr, "/device:%s:0", mctxs[0]->model->devicestr);
    } else {
        sprintf(tf_devicestr, "/device:%s", mctxs[0]->model->devicestr);
    }

    for (size_t i = 0; i < ninputs; ++i) {
        RAI_Tensor *batched_input_tensors[nbatches];

        for (size_t b = 0; b < nbatches; ++b) {
            batched_input_tensors[b] = mctxs[b]->inputs[i].tensor;
        }

        if (nbatches > 1) {
            RAI_Tensor *batched_tensor =
                RAI_TensorCreateByConcatenatingTensors(batched_input_tensors, nbatches);
            inputTensorsHandles[i] =
                TFE_HandleFromDLPack(batched_tensor, status, mctxs[0]->model->session);
        } else {
            inputTensorsHandles[i] =
                TFE_HandleFromDLPack(batched_input_tensors[0], status, mctxs[0]->model->session);
        }

        if (TF_GetCode(status) != TF_OK) {
            char *errorMessage = RedisModule_Strdup(TF_Message(status));
            RAI_SetError(error, RAI_EMODELRUN, errorMessage);
            TF_DeleteStatus(status);
            RedisModule_Free(errorMessage);
            return 1;
        }

        if (on_cpu) {
            deviceInputTensorsHandles[i] = inputTensorsHandles[i];
        }
        else {
            deviceInputTensorsHandles[i] = TFE_TensorHandleCopyToDevice(
                inputTensorsHandles[i], mctxs[0]->model->session, tf_devicestr, status);
        }

        if (TF_GetCode(status) != TF_OK) {
            char *errorMessage = RedisModule_Strdup(TF_Message(status));
            RAI_SetError(error, RAI_EMODELRUN, errorMessage);
            TF_DeleteStatus(status);
            RedisModule_Free(errorMessage);
            return 1;
        }
    }

    TFE_Op *fn_op = TFE_NewOp(mctxs[0]->model->session, RAI_TF_FN_NAME, status);
    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELRUN, errorMessage);
        TF_DeleteStatus(status);
        RedisModule_Free(errorMessage);
        return 1;
    }

    TFE_OpAddInputList(fn_op, deviceInputTensorsHandles, ninputs, status);
    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELRUN, errorMessage);
        TF_DeleteStatus(status);
        RedisModule_Free(errorMessage);
        return 1;
    }

    int noutputs_ = noutputs;
    TFE_Execute(fn_op, deviceOutputTensorsHandles, &noutputs_, status);
    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELRUN, errorMessage);
        TF_DeleteStatus(status);
        RedisModule_Free(errorMessage);
        return 1;
    }

    for (size_t i = 0; i < ninputs; ++i) {
        TFE_DeleteTensorHandle(inputTensorsHandles[i]);
        if (!on_cpu) {
            TFE_DeleteTensorHandle(deviceInputTensorsHandles[i]);
        }
    }

    if (TF_GetCode(status) != TF_OK) {
        char *errorMessage = RedisModule_Strdup(TF_Message(status));
        RAI_SetError(error, RAI_EMODELRUN, errorMessage);
        TF_DeleteStatus(status);
        RedisModule_Free(errorMessage);
        return 1;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        if (on_cpu) {
            outputTensorsHandles[i] = deviceOutputTensorsHandles[i];
        }
        else {
            outputTensorsHandles[i] = TFE_TensorHandleCopyToDevice(
                deviceOutputTensorsHandles[i], mctxs[0]->model->session, "/device:CPU:0", status);
        }

        RAI_Tensor *outputTensor =
            RAI_TensorCreateFromDLTensor(TFE_HandleToDLPack(outputTensorsHandles[i], status));

        if (TF_GetCode(status) != TF_OK) {
            char *errorMessage = RedisModule_Strdup(TF_Message(status));
            RAI_SetError(error, RAI_EMODELRUN, errorMessage);
            TF_DeleteStatus(status);
            RedisModule_Free(errorMessage);
            return 1;
        }

        if (nbatches > 1) {
            if (RAI_TensorNumDims(outputTensor) == 0) {
                continue;
            }
            if (RAI_TensorDim(outputTensor, 0) != total_batch_size) {
                RAI_TensorFree(outputTensor);
                TF_DeleteStatus(status);
                RAI_SetError(error, RAI_EMODELRUN,
                             "ERR Model did not generate the expected batch size");
                return 1;
            }

            for (size_t b = 0; b < nbatches; b++) {
                mctxs[b]->outputs[i].tensor =
                    RAI_TensorCreateBySlicingTensor(outputTensor, batch_offsets[b], batch_sizes[b]);
            }
        } else {
            mctxs[0]->outputs[i].tensor = RAI_TensorGetShallowCopy(outputTensor);
        }
        RAI_TensorFree(outputTensor);
        if (!on_cpu) {
            TFE_DeleteTensorHandle(deviceOutputTensorsHandles[i]);
        }
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
