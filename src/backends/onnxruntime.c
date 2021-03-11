#define REDISMODULE_MAIN
#include <cuda_provider_factory.h>
#include "backends/onnxruntime.h"
#include "backends/util.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include <stdatomic.h>

#include "onnxruntime_c_api.h"

// Use as a wrapper for ORT api call. If ORT api hasn't returned null, it has failed.
// A label "error" must exist in every function that uses this macro.
#define ONNX_VALIDATE_STATUS(x)                                                                    \
    if ((status = (x)) != NULL)                                                                    \
        goto error;

OrtEnv *env = NULL;

// For model that run on GPU, onnx will not use the custom allocator (redis allocator), but
// the onnx allocator for GPU. But for the auxilery allocations of the input and output names,
// we will use the custom global allocator for models that run on GPU as well
OrtAllocator *global_allocator = NULL;
unsigned long long OnnxMemory = 0;
unsigned long long OnnxMemoryAccessCounter = 0;

const OrtMemoryInfo *AllocatorInfo(const OrtAllocator *allocator) {
    (void)allocator;
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    OrtMemoryInfo *mem_info;
    if (ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mem_info) != NULL) {
        return NULL;
    }
    return mem_info;
}

// Allocate address with 64-byte alignment to cope with onnx optimizations.
void *AllocatorAlloc(OrtAllocator *ptr, size_t size) {

    (void)ptr;
    // Allocate an additional 63 bytes to ensure that we can return an address which is
    // 64-byte aligned, and an additional space in the size of a pointer to store
    // the address that RedisModule_Alloc returns.
    int offset = 63 + sizeof(void *);
    void *allocated_address = (void *)RedisModule_Alloc(size + offset);
    size_t allocated_size = RedisModule_MallocSize(allocated_address);
    // Update the total number of bytes that onnx is using and the number of accesses
    // that onnx made to the allocator.
    atomic_fetch_add(&OnnxMemory, allocated_size);
    atomic_fetch_add(&OnnxMemoryAccessCounter, 1);
    // This operation guarantees that p2 is the closest 64-aligned address to (p1+size_t).
    void **aligned_address = (void **)(((size_t)(allocated_address) + offset) & (~63));
    // This stores the address p1 right before p2 (so we can retrieve it when we free).
    aligned_address[-1] = allocated_address;
    return aligned_address;
}

void AllocatorFree(OrtAllocator *ptr, void *aligned_address) {
    (void)ptr;
    if (aligned_address == NULL) {
        return;
    }
    // Retrieve the address that we originally received from RedisModule_Alloc
    // (this is the address that we need to sent to RedisModule_Free).
    void *allocated_address = ((void **)aligned_address)[-1];
    size_t allocated_size = RedisModule_MallocSize(allocated_address);
    // Update the total number of bytes that onnx is using and the number of accesses
    // that onnx made to the allocator.
    atomic_fetch_sub(&OnnxMemory, allocated_size);
    atomic_fetch_add(&OnnxMemoryAccessCounter, 1);
    return RedisModule_Free(allocated_address);
}

unsigned long long RAI_GetMemoryInfoORT() { return OnnxMemory; }

unsigned long long RAI_GetMemoryAccessORT() { return OnnxMemoryAccessCounter; }

int RAI_InitBackendORT(int (*get_api_fn)(const char *, void *)) {
    get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
    get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
    get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
    get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
    get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));
    get_api_fn("RedisModule_Log", ((void **)&RedisModule_Log));
    get_api_fn("RedisModule_GetThreadSafeContext", ((void **)&RedisModule_GetThreadSafeContext));
    get_api_fn("RedisModule_FreeThreadSafeContext", ((void **)&RedisModule_FreeThreadSafeContext));
    get_api_fn("RedisModule_MallocSize", ((void **)&RedisModule_MallocSize));
    return REDISMODULE_OK;
}

bool setDeviceId(const char *devicestr, OrtSessionOptions *session_options, RAI_Error *error) {

    RAI_Device device;
    int64_t deviceid;
    if (!parseDeviceStr(devicestr, &device, &deviceid)) {
        RAI_SetError(error, RAI_EMODELCREATE, "ERR unsupported device");
        return false;
    }

    // TODO: we will need to propose a more dynamic way to request a specific provider,
    // e.g. given the name, in ONNXRuntime
#if RAI_ONNXRUNTIME_USE_CUDA
    if (device == RAI_DEVICE_GPU) {
        if (deviceid == -1) {
            // ORT does not like device id as -1
            deviceid = 0;
        }
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, deviceid);
    }
#else
    // TODO: Do dynamic device/provider check with GetExecutionProviderType or something on these
    // lines
    if (device == RAI_DEVICE_GPU) {
        RAI_SetError(error, RAI_EMODELCREATE, "ERR GPU requested but ONNX couldn't find CUDA");
        return false;
    }
#endif
    return true;
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
    } else if (dtype.code == kDLInt) {
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
    } else if (dtype.code == kDLUInt) {
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
        return (DLDataType){.code = kDLFloat, .bits = 32, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return (DLDataType){.code = kDLInt, .bits = 8, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return (DLDataType){.code = kDLInt, .bits = 16, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return (DLDataType){.code = kDLInt, .bits = 64, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return (DLDataType){.code = kDLUInt, .bits = 8, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return (DLDataType){.code = kDLUInt, .bits = 16, .lanes = 1};
    default:
        return (DLDataType){.bits = 0};
    }
}

OrtValue *RAI_OrtValueFromTensors(RAI_Tensor **ts, size_t count, RAI_Error *error) {
    OrtStatus *status = NULL;
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);

    size_t batch_size = 0;
    size_t batch_byte_size = 0;

    for (size_t i = 0; i < count; i++) {
        batch_size += ts[i]->tensor.dl_tensor.shape[0];
        batch_byte_size += RAI_TensorByteSize(ts[i]);
    }

    RAI_Tensor *t0 = ts[0];
    const int ndim = t0->tensor.dl_tensor.ndim;
    int64_t batched_shape[ndim];
    for (size_t i = 1; i < ndim; i++) {
        batched_shape[i] = t0->tensor.dl_tensor.shape[i];
    }
    batched_shape[0] = batch_size;

    OrtValue *out;
    if (count > 1) {
        ONNX_VALIDATE_STATUS(
            ort->CreateTensorAsOrtValue(global_allocator, batched_shape, t0->tensor.dl_tensor.ndim,
                                        RAI_GetOrtDataTypeFromDL(t0->tensor.dl_tensor.dtype), &out))

        char *ort_data;
        ONNX_VALIDATE_STATUS(ort->GetTensorMutableData(out, (void **)&ort_data))
        size_t offset = 0;
        for (size_t i = 0; i < count; i++) {
            memcpy(ort_data + offset, RAI_TensorData(ts[i]), RAI_TensorByteSize(ts[i]));
            offset += RAI_TensorByteSize(ts[i]);
        }
    } else {
        ONNX_VALIDATE_STATUS(ort->CreateTensorWithDataAsOrtValue(
            global_allocator->Info(global_allocator), t0->tensor.dl_tensor.data,
            RAI_TensorByteSize(t0), t0->tensor.dl_tensor.shape, t0->tensor.dl_tensor.ndim,
            RAI_GetOrtDataTypeFromDL(t0->tensor.dl_tensor.dtype), &out))
    }
    return out;

error:
    RAI_SetError(error, RAI_EMODELRUN, ort->GetErrorMessage(status));
    ort->ReleaseStatus(status);
    return NULL;
}

RAI_Tensor *RAI_TensorCreateFromOrtValue(OrtValue *v, size_t batch_offset, long long batch_size,
                                         RAI_Error *error) {
    OrtStatus *status = NULL;
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    RAI_Tensor *ret = NULL;
    int64_t *shape = NULL;
    int64_t *strides = NULL;

    int is_tensor;
    ONNX_VALIDATE_STATUS(ort->IsTensor(v, &is_tensor))
    if (!is_tensor) {
        // TODO: if not tensor, flatten the data structure (sequence or map) and store it in a
        // tensor. If return value is string, emit warning.
        return NULL;
    }

    ret = RAI_TensorNew();
    DLDevice device = (DLDevice){.device_type = kDLCPU, .device_id = 0};

    OrtTensorTypeAndShapeInfo *info;
    ONNX_VALIDATE_STATUS(ort->GetTensorTypeAndShape(v, &info))

    {
        size_t ndims;
        ONNX_VALIDATE_STATUS(ort->GetDimensionsCount(info, &ndims))
        int64_t dims[ndims];
        ONNX_VALIDATE_STATUS(ort->GetDimensions(info, dims, ndims))
        enum ONNXTensorElementDataType ort_dtype;
        ONNX_VALIDATE_STATUS(ort->GetTensorElementType(info, &ort_dtype))
        int64_t total_batch_size = dims[0];
        total_batch_size = total_batch_size > 0 ? total_batch_size : 1;

        shape = RedisModule_Calloc(ndims, sizeof(*shape));
        strides = RedisModule_Calloc(ndims, sizeof(*strides));
        for (int64_t i = 0; i < ndims; ++i) {
            shape[i] = dims[i];
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

        DLDataType dtype = RAI_GetDLDataTypeFromORT(ort_dtype);
#ifdef RAI_COPY_RUN_OUTPUT
        char *ort_data;
        ONNX_VALIDATE_STATUS(ort->GetTensorMutableData(v, (void **)&ort_data))
        size_t elem_count;
        ONNX_VALIDATE_STATUS(ort->GetTensorShapeElementCount(info, &elem_count))

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

        ret->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.device = device,
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

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                              const char *modeldef, size_t modellen, RAI_Error *error) {

    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    char **inputs_ = NULL;
    char **outputs_ = NULL;
    OrtSessionOptions *session_options = NULL;
    OrtSession *session = NULL;
    OrtStatus *status = NULL;

    // In the first time we set a model for onnx, we create an environment and register
    // an allocator to it that uses Redis allocator. This allocator is going to be used for
    // allocating buffers when creating and running models that run on CPU, and for allocations of
    // models inputs and outputs names (for both models that run on CPU and GPU)
    if (env == NULL) {
        ONNX_VALIDATE_STATUS(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env))
        ONNX_VALIDATE_STATUS(ort->CreateCustomDeviceAllocator(
            ORT_API_VERSION, AllocatorAlloc, AllocatorFree, AllocatorInfo, &global_allocator))
        ONNX_VALIDATE_STATUS(ort->RegisterCustomDeviceAllocator(env, global_allocator))
    }

    ONNX_VALIDATE_STATUS(ort->CreateSessionOptions(&session_options))
    if (strcasecmp(devicestr, "CPU") == 0) {
        // These are required to ensure that onnx will use the registered REDIS allocator (for
        // a model that defined to run on CPU).
        ONNX_VALIDATE_STATUS(
            ort->AddSessionConfigEntry(session_options, "session.use_env_allocators", "1"))
        ONNX_VALIDATE_STATUS(ort->DisableCpuMemArena(session_options))
    }

    // TODO: these options could be configured at the AI.CONFIG level
    ONNX_VALIDATE_STATUS(ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC))
    ONNX_VALIDATE_STATUS(
        ort->SetIntraOpNumThreads(session_options, (int)opts.backends_intra_op_parallelism))
    ONNX_VALIDATE_STATUS(
        ort->SetInterOpNumThreads(session_options, (int)opts.backends_inter_op_parallelism))

    // If the model is set for GPU, this will set CUDA provider for the session,
    // so that onnx will use its own allocator for CUDA (not Redis allocator)
    if (!setDeviceId(devicestr, session_options, error)) {
        ort->ReleaseSessionOptions(session_options);
        return NULL;
    }

    ONNX_VALIDATE_STATUS(
        ort->CreateSessionFromArray(env, modeldef, modellen, session_options, &session))

    size_t n_input_nodes;
    ONNX_VALIDATE_STATUS(ort->SessionGetInputCount(session, &n_input_nodes))
    size_t n_output_nodes;
    ONNX_VALIDATE_STATUS(ort->SessionGetOutputCount(session, &n_output_nodes))

    inputs_ = array_new(char *, n_input_nodes);
    for (long long i = 0; i < n_input_nodes; i++) {
        char *input_name;
        ONNX_VALIDATE_STATUS(ort->SessionGetInputName(session, i, global_allocator, &input_name))
        inputs_ = array_append(inputs_, input_name);
    }

    outputs_ = array_new(char *, n_output_nodes);
    for (long long i = 0; i < n_output_nodes; i++) {
        char *output_name;
        ONNX_VALIDATE_STATUS(ort->SessionGetOutputName(session, i, global_allocator, &output_name))
        outputs_ = array_append(outputs_, output_name);
    }

    // Since ONNXRuntime doesn't have a re-serialization function,
    // we cache the blob in order to re-serialize it.
    // Not optimal for storage purposes, but again, it may be temporary
    char *buffer = RedisModule_Calloc(modellen, sizeof(*buffer));
    memcpy(buffer, modeldef, modellen);

    RAI_Model *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->model = NULL;
    ret->session = session;
    ret->backend = backend;
    ret->devicestr = RedisModule_Strdup(devicestr);
    ret->refCount = 1;
    ret->opts = opts;
    ret->data = buffer;
    ret->datalen = modellen;
    ret->ninputs = n_input_nodes;
    ret->noutputs = n_output_nodes;
    ret->inputs = inputs_;
    ret->outputs = outputs_;

    return ret;

error:
    RAI_SetError(error, RAI_EMODELCREATE, ort->GetErrorMessage(status));
    if (session_options) {
        ort->ReleaseSessionOptions(session_options);
    }
    if (inputs_) {
        n_input_nodes = array_len(inputs_);
        for (uint32_t i = 0; i < n_input_nodes; i++) {
            status = ort->AllocatorFree(global_allocator, inputs_[i]);
        }
        array_free(inputs_);
    }
    if (outputs_) {
        n_output_nodes = array_len(outputs_);
        for (uint32_t i = 0; i < n_output_nodes; i++) {
            status = ort->AllocatorFree(global_allocator, outputs_[i]);
        }
        array_free(outputs_);
    }
    if (session) {
        ort->ReleaseSession(session);
    }
    ort->ReleaseStatus(status);
    return NULL;
}

void RAI_ModelFreeORT(RAI_Model *model, RAI_Error *error) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    OrtStatus *status = NULL;

    for (uint32_t i = 0; i < model->ninputs; i++) {
        ONNX_VALIDATE_STATUS(ort->AllocatorFree(global_allocator, model->inputs[i]))
    }
    array_free(model->inputs);

    for (uint32_t i = 0; i < model->noutputs; i++) {
        ONNX_VALIDATE_STATUS(ort->AllocatorFree(global_allocator, model->outputs[i]))
    }
    array_free(model->outputs);

    RedisModule_Free(model->devicestr);
    RedisModule_Free(model->data);
    ort->ReleaseSession(model->session);
    model->model = NULL;
    model->session = NULL;
    return;

error:
    RAI_SetError(error, RAI_EMODELFREE, ort->GetErrorMessage(status));
    ort->ReleaseStatus(status);
}

int RAI_ModelRunORT(RAI_ModelRunCtx **mctxs, RAI_Error *error) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);

    OrtSession *session = mctxs[0]->model->session;
    if (session == NULL) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR ONNXRuntime session was not allocated");
        return REDISMODULE_ERR;
    }

    const size_t nbatches = array_len(mctxs);
    if (nbatches == 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR No batches to run");
        return REDISMODULE_ERR;
    }

    size_t batch_sizes[nbatches];
    size_t batch_offsets[nbatches];
    size_t total_batch_size = 0;
    if (array_len(mctxs[0]->inputs) > 0) {
        for (size_t b = 0; b < nbatches; ++b) {
            batch_sizes[b] = RAI_TensorDim(mctxs[b]->inputs[0].tensor, 0);
            total_batch_size += batch_sizes[b];
        }
        batch_offsets[0] = 0;
        for (size_t b = 1; b < nbatches; ++b) {
            batch_offsets[b] = batch_offsets[b - 1] + batch_sizes[b - 1];
        }
    }

    OrtStatus *status = NULL;
    size_t n_input_nodes;
    ONNX_VALIDATE_STATUS(ort->SessionGetInputCount(session, &n_input_nodes))

    size_t n_output_nodes;
    ONNX_VALIDATE_STATUS(ort->SessionGetOutputCount(session, &n_output_nodes)) {
        const char *input_names[n_input_nodes];
        const char *output_names[n_output_nodes];

        OrtValue *inputs[n_input_nodes];
        OrtValue *outputs[n_output_nodes];

        const size_t ninputs = array_len(mctxs[0]->inputs);
        const size_t noutputs = array_len(mctxs[0]->outputs);

        if (ninputs != n_input_nodes) {
            char msg[70];
            sprintf(msg, "ERR Expected %li inputs but got %li", n_input_nodes, ninputs);
            RAI_SetError(error, RAI_EMODELRUN, msg);
            return REDISMODULE_ERR;
        }

        if (noutputs != n_output_nodes) {
            char msg[70];
            sprintf(msg, "ERR Expected %li outputs but got %li", n_output_nodes, noutputs);
            RAI_SetError(error, RAI_EMODELRUN, msg);
            return REDISMODULE_ERR;
        }

        for (size_t i = 0; i < n_input_nodes; i++) {
            char *input_name;
            ONNX_VALIDATE_STATUS(
                ort->SessionGetInputName(session, i, global_allocator, &input_name))
            input_names[i] = input_name;

            RAI_Tensor *batched_input_tensors[nbatches];
            for (size_t b = 0; b < nbatches; b++) {
                batched_input_tensors[b] = mctxs[b]->inputs[i].tensor;
            }

            inputs[i] = RAI_OrtValueFromTensors(batched_input_tensors, nbatches, error);
            if (error->code != RAI_OK) {
                ort->ReleaseStatus(status);
                return REDISMODULE_ERR;
            }
        }

        for (size_t i = 0; i < n_output_nodes; i++) {
            char *output_name;
            ONNX_VALIDATE_STATUS(
                ort->SessionGetOutputName(session, i, global_allocator, &output_name))
            output_names[i] = output_name;
            outputs[i] = NULL;
        }

        OrtRunOptions *run_options = NULL;
        ONNX_VALIDATE_STATUS(ort->Run(session, run_options, input_names,
                                      (const OrtValue *const *)inputs, n_input_nodes, output_names,
                                      n_output_nodes, outputs))

        for (size_t i = 0; i < n_output_nodes; i++) {
            if (nbatches > 1) {
                OrtTensorTypeAndShapeInfo *info;
                ONNX_VALIDATE_STATUS(ort->GetTensorTypeAndShape(outputs[i], &info))
                size_t ndims;
                ONNX_VALIDATE_STATUS(ort->GetDimensionsCount(info, &ndims))
                int64_t dims[ndims];
                ONNX_VALIDATE_STATUS(ort->GetDimensions(info, dims, ndims))
                if (dims[0] != total_batch_size) {
                    RAI_SetError(error, RAI_EMODELRUN,
                                 "ERR Model did not generate the expected batch size");
                    ort->ReleaseStatus(status);
                    return REDISMODULE_ERR;
                }

                for (size_t b = 0; b < nbatches; b++) {
                    RAI_Tensor *output_tensor = RAI_TensorCreateFromOrtValue(
                        outputs[i], batch_offsets[b], batch_sizes[b], error);
                    if (error->code != RAI_OK) {
                        ort->ReleaseStatus(status);
                        return REDISMODULE_ERR;
                    }
                    if (output_tensor) {
                        mctxs[b]->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
                        RAI_TensorFree(output_tensor);
                    } else {
                        printf("ERR: non-tensor output from ONNX models, ignoring (currently "
                               "unsupported)");
                    }
                }
            } else {
                RAI_Tensor *output_tensor = RAI_TensorCreateFromOrtValue(outputs[i], 0, -1, error);
                if (error->code != RAI_OK) {
                    ort->ReleaseStatus(status);
                    return REDISMODULE_ERR;
                }
                if (output_tensor) {
                    mctxs[0]->outputs[i].tensor = RAI_TensorGetShallowCopy(output_tensor);
                    RAI_TensorFree(output_tensor);
                } else {
                    printf("ERR: non-tensor output from ONNX models, ignoring (currently "
                           "unsupported)");
                }
            }
            ort->ReleaseValue(outputs[i]);
        }
        for (size_t i = 0; i < n_input_nodes; i++) {
            ort->ReleaseValue(inputs[i]);
        }
        return REDISMODULE_OK;
    }

error:
    RAI_SetError(error, RAI_EMODELRUN, ort->GetErrorMessage(status));
    ort->ReleaseStatus(status);
    return REDISMODULE_ERR;
}

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
    *buffer = RedisModule_Calloc(model->datalen, sizeof(char));
    memcpy(*buffer, model->data, model->datalen);
    *len = model->datalen;

    return REDISMODULE_OK;
}

const char *RAI_GetBackendVersionORT(void) { return OrtGetApiBase()->GetVersionString(); }
