#define REDISMODULE_MAIN
#include <cuda_provider_factory.h>
#include "backends/util.h"
#include <stdatomic.h>
#include <backends/onnx_timeout.h>
#include <pthread.h>
#include "util/arr.h"
#include "backends/onnxruntime.h"
#include "redis_ai_objects/tensor.h"

#include "onnxruntime_c_api.h"
#include "backends_api.h"

// Use as a wrapper for ORT api call. If ORT api hasn't returned null, it has failed.
// A label "error" must exist in every function that uses this macro.
#define ONNX_VALIDATE_STATUS(x)                                                                    \
    if ((status = (x)) != NULL)                                                                    \
        goto error;

OrtEnv *env = NULL;

// For model that run on GPU, onnx will not use the custom allocator (redis allocator), but
// the onnx allocator for GPU. But for the auxiliary allocations of the input and output names,
// we will use the custom global allocator for models that run on GPU as well.
OrtMemoryInfo *mem_info = NULL;
OrtAllocator *global_allocator = NULL;
unsigned long long OnnxMemory = 0;
unsigned long long OnnxMemoryAccessCounter = 0;

const OrtMemoryInfo *AllocatorInfo(const OrtAllocator *allocator) {
    (void)allocator;
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    if (mem_info != NULL) {
        return mem_info;
    }
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

int RAI_InitBackendORT(int (*get_api_fn)(const char *, void **)) {
    // Export redis callbacks.
    get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
    get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
    get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
    get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
    get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));
    get_api_fn("RedisModule_Log", ((void **)&RedisModule_Log));
    get_api_fn("RedisModule_GetThreadSafeContext", ((void **)&RedisModule_GetThreadSafeContext));
    get_api_fn("RedisModule_FreeThreadSafeContext", ((void **)&RedisModule_FreeThreadSafeContext));
    get_api_fn("RedisModule_MallocSize", ((void **)&RedisModule_MallocSize));

    // Export RedisAI callbacks.
    get_api_fn("GetThreadId", ((void **)&RedisAI_GetThreadId));
    get_api_fn("GetNumThreadsPerQueue", ((void **)&RedisAI_GetNumThreadsPerQueue));
    get_api_fn("GetModelExecutionTimeout", ((void **)&RedisAI_GetModelExecutionTimeout));
    get_api_fn("GetThreadsCount", ((void **)&RedisAI_GetThreadsCount));

    // Create a global array of onnx runSessions, with an entry for every working thread.
    RAI_InitGlobalRunSessionsORT();

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
    } else if (dtype.code == kDLBool) {
        switch (dtype.bits) {
        case 8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        }
    } else if (dtype.code == kDLString) {
        switch (dtype.bits) {
        case 8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
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
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return (DLDataType){.code = kDLString, .bits = 8, .lanes = 1};
    default:
        return (DLDataType){.bits = 0};
    }
}

int RAI_OrtValueFromTensors(RAI_Tensor **ts, size_t count, OrtValue **input,
                            OrtStatus **status_ptr) {
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
    size_t batched_tensor_len = 1;
    for (size_t i = 1; i < ndim; i++) {
        batched_shape[i] = t0->tensor.dl_tensor.shape[i];
        batched_tensor_len *= batched_shape[i];
    }
    batched_shape[0] = batch_size;
    batched_tensor_len *= batch_size;

    OrtValue *out;
    if (t0->tensor.dl_tensor.dtype.code == kDLString) {
        ONNX_VALIDATE_STATUS(
                ort->CreateTensorAsOrtValue(global_allocator, batched_shape, t0->tensor.dl_tensor.ndim,
                                            RAI_GetOrtDataTypeFromDL(t0->tensor.dl_tensor.dtype), &out));
        size_t element_index = 0;
        for (size_t i = 0; i < count; i++) {
            // go over all strings stored in the tensors' data from all tensors and set them in the ORT tensor.
            uint64_t *offsets = ts[i]->tensor.dl_tensor.elements_length;
            char str_element0[offsets[0]];
            strncpy(str_element0, RAI_TensorData(ts[i]), offsets[0]);
            ONNX_VALIDATE_STATUS(ort->FillStringTensorElement(out, str_element0 ,element_index++));
            for (size_t j = 1; j < RAI_TensorLength(ts[i]); j++) {
                char str_element[offsets[j]-offsets[j-1]];
                strncpy(str_element, RAI_TensorData(ts[i]) + offsets[j-1], offsets[j]-offsets[j-1]);
                ONNX_VALIDATE_STATUS(ort->FillStringTensorElement(out, str_element ,element_index++));
            }
        }
    } else if (count > 1) {
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
    *input = out;
    return REDISMODULE_OK;

error:
    *status_ptr = status;
    return REDISMODULE_ERR;
}

RAI_Tensor *RAI_TensorCreateFromOrtValue(OrtValue *v, size_t batch_offset, long long batch_size,
                                         RAI_Error *error) {
    OrtStatus *status = NULL;
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    RAI_Tensor *output_tensor = NULL;
    OrtTensorTypeAndShapeInfo *info = NULL;

    int is_tensor;
    ONNX_VALIDATE_STATUS(ort->IsTensor(v, &is_tensor))
    if (!is_tensor) {
        // TODO: if not tensor, flatten the data structure (sequence or map) and store it in a tensor.
        return NULL;
    }

    // Create an empty RAI_Tensor based on the tensor's meat-data
    ONNX_VALIDATE_STATUS(ort->GetTensorTypeAndShape(v, &info))
    ONNXTensorElementDataType ort_dtype;
    ONNX_VALIDATE_STATUS(ort->GetTensorElementType(info, &ort_dtype))
    DLDataType data_type = RAI_GetDLDataTypeFromORT(ort_dtype);
    size_t data_type_size = data_type.bits / 8;
    int64_t total_batch_size;
    {
        size_t n_dims;
        ONNX_VALIDATE_STATUS(ort->GetDimensionsCount(info, &n_dims))
        int64_t dims[n_dims];
        ONNX_VALIDATE_STATUS(ort->GetDimensions(info, dims, n_dims))
        total_batch_size = dims[0];

        // if batch size is non-negative (meaning we ran a batch), we restore the
        // original first dimension of this particular tensor.
        if (batch_size > 0) {
            dims[0] = batch_size;
        } else {
            batch_size = total_batch_size; // = dims[0]
        }
        output_tensor = RAI_TensorNew(data_type, data_type_size, (const long long *)dims, (int)n_dims);
    }

    size_t output_tensor_len = RAI_TensorLength(output_tensor);
    size_t elem_count;
    ONNX_VALIDATE_STATUS(ort->GetTensorShapeElementCount(info, &elem_count))
    RedisModule_Assert(elem_count / total_batch_size * batch_size == output_tensor_len);

    if (data_type.code != kDLString) {
        char *ort_data;
        ONNX_VALIDATE_STATUS(ort->GetTensorMutableData(v, (void **)&ort_data))

        size_t total_byte_size = elem_count * data_type_size;
        size_t sample_byte_size = total_byte_size / total_batch_size;
        memcpy(RAI_TensorData(output_tensor), ort_data + batch_offset * sample_byte_size, RAI_TensorByteSize(output_tensor));
    } else {
        // Calculate the blob size for this tensor and allocate space for it
        size_t element_index = output_tensor_len / batch_size;
        size_t blob_len = 0;
        for (size_t i = 0; i < output_tensor_len; i++) {
            size_t str_element_len;
            ONNX_VALIDATE_STATUS(ort->GetStringTensorElementLength(v, element_index++, &str_element_len));
            blob_len += str_element_len + 1; // Add space for null character at the end of every string
            element_index++;
        }
        output_tensor->blobSize = blob_len;
        output_tensor->tensor.dl_tensor.data = RedisModule_Alloc(blob_len);

        // Go over again and copy tensor data elements one by one
        element_index = output_tensor_len / batch_size;
        uint64_t *offsets = RAI_TensorStringElementsOffsets(output_tensor);
        offsets[0] = 0;
        for (size_t i = 0; i < output_tensor_len; i++) {
            size_t str_element_len;
            ONNX_VALIDATE_STATUS(ort->GetStringTensorElementLength(v, element_index, &str_element_len));
            char string_data[str_element_len + 1];
            ONNX_VALIDATE_STATUS(ort->GetStringTensorElement(v, element_index, str_element_len, string_data));
            string_data[str_element_len] = '\0';
            memcpy(RAI_TensorData(output_tensor) + offsets[i], string_data, str_element_len+1);
            if (i<output_tensor_len) offsets[i+1] = offsets[i]+str_element_len+1;
        }
    }

    ort->ReleaseTensorTypeAndShapeInfo(info);
    return output_tensor;

error:
    RAI_SetError(error, RAI_EMODELCREATE, ort->GetErrorMessage(status));
    ort->ReleaseStatus(status);
    if (output_tensor != NULL) {
        RAI_TensorFree(output_tensor);
    }
    if (info != NULL) {
        ort->ReleaseTensorTypeAndShapeInfo(info);
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
    ort->ReleaseSessionOptions(session_options);

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

int RAI_ModelRunORT(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);

    OrtSession *session = RAI_ModelGetSession(model);
    if (session == NULL) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR ONNXRuntime session was not allocated");
        return REDISMODULE_ERR;
    }

    const size_t n_batches = array_len(ectxs);
    if (n_batches == 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR No batches to run");
        return REDISMODULE_ERR;
    }

    size_t batch_sizes[n_batches];
    size_t batch_offsets[n_batches];
    size_t total_batch_size = 0;
    const size_t n_inputs = RAI_ExecutionCtx_NumInputs(ectxs[0]);
    const size_t n_outputs = RAI_ExecutionCtx_NumOutputs(ectxs[0]);
    if (n_batches > 0) {
        for (size_t b = 0; b < n_batches; ++b) {
            batch_sizes[b] = RAI_TensorDim(RAI_ExecutionCtx_GetInput(ectxs[b], 0), 0);
            total_batch_size += batch_sizes[b];
        }
        batch_offsets[0] = 0;
        for (size_t b = 1; b < n_batches; ++b) {
            batch_offsets[b] = batch_offsets[b - 1] + batch_sizes[b - 1];
        }
    }

    OrtStatus *status = NULL;
    array_new_on_stack(const char *, 5, input_names);
    array_new_on_stack(const char *, 5, output_names);
    array_new_on_stack(OrtValue *, 5, inputs);
    array_new_on_stack(OrtValue *, 5, outputs);
    OrtRunOptions *run_options = NULL;
    long run_session_index;
    OrtTensorTypeAndShapeInfo *info = NULL;
    {
        size_t n_input_nodes;
        size_t n_output_nodes;
        ONNX_VALIDATE_STATUS(ort->SessionGetInputCount(session, &n_input_nodes))
        ONNX_VALIDATE_STATUS(ort->SessionGetOutputCount(session, &n_output_nodes))

        if (n_inputs != n_input_nodes) {
            char msg[70];
            sprintf(msg, "ERR Expected %li inputs but got %li", n_input_nodes, n_inputs);
            RAI_SetError(error, RAI_EMODELRUN, msg);
            return REDISMODULE_ERR;
        }

        if (n_outputs != n_output_nodes) {
            char msg[70];
            sprintf(msg, "ERR Expected %li outputs but got %li", n_output_nodes, n_outputs);
            RAI_SetError(error, RAI_EMODELRUN, msg);
            return REDISMODULE_ERR;
        }

        for (size_t i = 0; i < n_input_nodes; i++) {
            char *input_name;
            ONNX_VALIDATE_STATUS(
                ort->SessionGetInputName(session, i, global_allocator, &input_name))
            input_names = array_append(input_names, input_name);

            RAI_Tensor *batched_input_tensors[n_batches];
            for (size_t b = 0; b < n_batches; b++) {
                batched_input_tensors[b] = RAI_ExecutionCtx_GetInput(ectxs[b], i);
            }
            OrtValue *input;
            if (RAI_OrtValueFromTensors(batched_input_tensors, n_batches, &input, &status) !=
                REDISMODULE_OK) {
                goto error;
            }
            inputs = array_append(inputs, input);
        }

        for (size_t i = 0; i < n_output_nodes; i++) {
            char *output_name;
            ONNX_VALIDATE_STATUS(
                ort->SessionGetOutputName(session, i, global_allocator, &output_name))
            output_names = array_append(output_names, output_name);
            outputs = array_append(outputs, NULL);
        }

        ONNX_VALIDATE_STATUS(ort->CreateRunOptions(&run_options));
        // Set the created run option in the global RunSessions and save its index.
        RAI_ActivateRunSessionCtxORT(run_options, &run_session_index);
        if (run_session_index == -1) {
            RAI_SetError(
                error, RAI_EMODELRUN,
                "Cannot execute onnxruntime model synchronously, use async execution instead");
            ort->ReleaseRunOptions(run_options);
            run_options = NULL;
            goto error;
        }

        ONNX_VALIDATE_STATUS(ort->Run(session, run_options, input_names,
                                      (const OrtValue *const *)inputs, n_input_nodes, output_names,
                                      n_output_nodes, outputs));
        RAI_ResetRunSessionCtxORT(run_session_index);
        run_options = NULL;

        for (uint32_t i = 0; i < n_inputs; i++) {
            status = ort->AllocatorFree(global_allocator, (void *)input_names[i]);
        }
        array_free(input_names);
        for (uint32_t i = 0; i < n_outputs; i++) {
            status = ort->AllocatorFree(global_allocator, (void *)output_names[i]);
        }
        array_free(output_names);

        for (size_t i = 0; i < n_output_nodes; i++) {
            if (n_batches > 1) {
                ONNX_VALIDATE_STATUS(ort->GetTensorTypeAndShape(outputs[i], &info))
                size_t ndims;
                ONNX_VALIDATE_STATUS(ort->GetDimensionsCount(info, &ndims))
                int64_t dims[ndims];
                ONNX_VALIDATE_STATUS(ort->GetDimensions(info, dims, ndims))
                ort->ReleaseTensorTypeAndShapeInfo(info);
                if (dims[0] != total_batch_size) {
                    RAI_SetError(error, RAI_EMODELRUN,
                                 "ERR Model did not generate the expected batch size");
                    goto error;
                }

                for (size_t b = 0; b < n_batches; b++) {
                    RAI_Tensor *output_tensor = RAI_TensorCreateFromOrtValue(
                        outputs[i], batch_offsets[b], batch_sizes[b], error);
                    if (error->code != RAI_OK) {
                        goto error;
                    }
                    if (output_tensor) {
                        RAI_ExecutionCtx_SetOutput(ectxs[b], output_tensor, i);
                    } else {
                        RedisModule_Log(NULL, "warning",
                                        "non-tensor output from ONNX models, ignoring (currently "
                                        "unsupported)");
                    }
                }
            } else {
                RAI_Tensor *output_tensor = RAI_TensorCreateFromOrtValue(outputs[i], 0, -1, error);
                if (RAI_GetErrorCode(error) != RAI_OK) {
                    ort->ReleaseValue(outputs[i]);
                    goto error;
                }
                if (output_tensor) {
                    RAI_ExecutionCtx_SetOutput(ectxs[0], output_tensor, i);
                } else {
                    RedisModule_Log(NULL, "warning",
                                    "non-tensor output from ONNX models, ignoring (currently "
                                    "unsupported)");
                }
            }
            ort->ReleaseValue(outputs[i]);
        }
        array_free(outputs);
        for (size_t i = 0; i < n_input_nodes; i++) {
            ort->ReleaseValue(inputs[i]);
        }
        array_free(inputs);
        return REDISMODULE_OK;
    }

error:
    if (status) {
        RAI_SetError(error, RAI_EMODELRUN, ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
    }
    for (uint32_t i = 0; i < array_len(input_names); i++) {
        status = ort->AllocatorFree(global_allocator, (void *)input_names[i]);
    }
    array_free(input_names);
    for (uint32_t i = 0; i < array_len(output_names); i++) {
        status = ort->AllocatorFree(global_allocator, (void *)output_names[i]);
    }
    array_free(output_names);
    for (size_t i = 0; i < array_len(inputs); i++) {
        ort->ReleaseValue(inputs[i]);
    }
    array_free(inputs);
    for (size_t i = 0; i < array_len(outputs); i++) {
        ort->ReleaseValue(outputs[i]);
    }
    array_free(outputs);
    if (info) {
        ort->ReleaseTensorTypeAndShapeInfo(info);
    }
    if (run_options) {
        RAI_ResetRunSessionCtxORT(run_session_index);
    }
    return REDISMODULE_ERR;
}

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {
    *buffer = RedisModule_Calloc(model->datalen, sizeof(char));
    memcpy(*buffer, model->data, model->datalen);
    *len = model->datalen;

    return REDISMODULE_OK;
}

const char *RAI_GetBackendVersionORT(void) { return OrtGetApiBase()->GetVersionString(); }
