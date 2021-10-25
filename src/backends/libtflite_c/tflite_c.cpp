#include <sstream>
#include <iostream>
#include "tflite_c.h"
#include "redismodule.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace {

static DLDataType getDLDataType(const TfLiteTensor *tensor) {
    DLDataType dtype;
    dtype.lanes = 1;
    switch (tensor->type) {
    case kTfLiteUInt8:
        dtype.bits = 8;
        dtype.code = DLDataTypeCode::kDLUInt;
        break;
    case kTfLiteInt64:
        dtype.bits = 64;
        dtype.code = DLDataTypeCode::kDLInt;
        break;
    case kTfLiteInt32:
        dtype.bits = 32;
        dtype.code = DLDataTypeCode::kDLInt;
        break;
    case kTfLiteInt16:
        dtype.bits = 16;
        dtype.code = DLDataTypeCode::kDLInt;
        break;
    case kTfLiteInt8:
        dtype.bits = 8;
        dtype.code = DLDataTypeCode::kDLInt;
        break;
    case kTfLiteFloat32:
        dtype.bits = 32;
        dtype.code = DLDataTypeCode::kDLFloat;
        break;
    case kTfLiteFloat16:
        // TODO: nope so far
        dtype.bits = 16;
        dtype.code = DLDataTypeCode::kDLFloat;
        break;
    case kTfLiteBool:
        dtype.bits = 8;
        dtype.code = DLDataTypeCode::kDLBool;
        break;
    default:
        break;
    }
    return dtype;
}

static DLDevice getDLDevice(const TfLiteTensor *tensor, const int64_t &device_id) {
    DLDevice device;
    device.device_id = device_id;
    device.device_type = DLDeviceType::kDLCPU;
    return device;
}

size_t dltensorBytes(DLManagedTensor *t) {
    int64_t *shape = t->dl_tensor.shape;
    size_t len = 1;
    for (size_t i = 0; i < t->dl_tensor.ndim; ++i) {
        len *= shape[i];
    }

    size_t bytes = len * t->dl_tensor.dtype.bits / 8;

    return bytes;
}

void copyToTfLiteTensor(std::shared_ptr<tflite::Interpreter> interpreter, int tflite_input,
                        DLManagedTensor *input) {
    TfLiteTensor *tensor = interpreter->tensor(tflite_input);
    size_t nbytes = dltensorBytes(input);
    DLDataType dltensor_type = input->dl_tensor.dtype;
    const char *type_mismatch_msg = "Input tensor type doesn't match the type expected"
                                    " by the model definition";

    switch (tensor->type) {
    case kTfLiteUInt8:
        if (dltensor_type.code != kDLUInt || dltensor_type.bits != 8) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<uint8_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt64:
        if (dltensor_type.code != kDLInt || dltensor_type.bits != 64) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<int64_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt32:
        if (dltensor_type.code != kDLInt || dltensor_type.bits != 32) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<int32_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt16:
        if (dltensor_type.code != kDLInt || dltensor_type.bits != 16) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<int16_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt8:
        if (dltensor_type.code != kDLInt || dltensor_type.bits != 8) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<int8_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteFloat32:
        if (dltensor_type.code != kDLFloat || dltensor_type.bits != 32) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<float>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteBool:
        if (dltensor_type.code != kDLBool || dltensor_type.bits != 8) {
            throw std::logic_error(type_mismatch_msg);
        }
        memcpy(interpreter->typed_tensor<bool>(tflite_input), input->dl_tensor.data, nbytes);
    case kTfLiteFloat16:
        throw std::logic_error("Float16 not currently supported as input tensor data type");
    default:
        throw std::logic_error("Unsupported input data type");
    }
}

void deleter(DLManagedTensor *arg) {
    delete[](uint8_t *) arg->dl_tensor.data;
    delete[] arg->dl_tensor.shape;
    delete[] arg->dl_tensor.strides;
    RedisModule_Free(arg);
}

DLManagedTensor *toManagedDLPack(std::shared_ptr<tflite::Interpreter> interpreter,
                                 int tflite_output) {
    TfLiteTensor *tensor = interpreter->tensor(tflite_output);

    TfLiteIntArray *output_dims = tensor->dims;

    DLDataType dtype = getDLDataType(tensor);

    int64_t device_id = 0;
    DLDevice device = getDLDevice(tensor, device_id);

    DLTensor dl_tensor = (DLTensor){.data = new uint8_t[tensor->bytes],
                                    .device = device,
                                    .ndim = output_dims->size,
                                    .dtype = dtype,
                                    .shape = new int64_t[output_dims->size],
                                    .strides = new int64_t[output_dims->size],
                                    .byte_offset = 0};

    for (size_t i = 0; i < output_dims->size; i++) {
        dl_tensor.shape[i] = output_dims->data[i];
        dl_tensor.strides[i] = 1;
    }

    for (int64_t i = dl_tensor.ndim - 2; i >= 0; --i) {
        dl_tensor.strides[i] *= dl_tensor.strides[i + 1] * dl_tensor.shape[i + 1];
    }

    auto output_size = output_dims->data[output_dims->size - 1];

    switch (tensor->type) {
    case kTfLiteUInt8:
        memcpy(dl_tensor.data, interpreter->typed_tensor<uint8_t>(tflite_output), tensor->bytes);
        break;
    case kTfLiteInt64:
        memcpy(dl_tensor.data, interpreter->typed_tensor<int64_t>(tflite_output), tensor->bytes);
        break;
    case kTfLiteInt32:
        memcpy(dl_tensor.data, interpreter->typed_tensor<int32_t>(tflite_output), tensor->bytes);
        break;
    case kTfLiteInt16:
        memcpy(dl_tensor.data, interpreter->typed_tensor<int16_t>(tflite_output), tensor->bytes);
        break;
    case kTfLiteInt8:
        memcpy(dl_tensor.data, interpreter->typed_tensor<int8_t>(tflite_output), tensor->bytes);
        break;
    case kTfLiteFloat32:
        memcpy(dl_tensor.data, interpreter->typed_tensor<float>(tflite_output), tensor->bytes);
        break;
    case kTfLiteBool:
        memcpy(dl_tensor.data, interpreter->typed_tensor<bool>(tflite_output), tensor->bytes);
        break;
    case kTfLiteFloat16:
        throw std::logic_error("Float16 not currently supported as output tensor data type");
    default:
        throw std::logic_error("Unsupported output data type");
    }

    // We use alloc here to allow deallocation from the module
    DLManagedTensor *output = (DLManagedTensor *)RedisModule_Alloc(sizeof(DLManagedTensor));
    output->dl_tensor = dl_tensor;
    output->manager_ctx = NULL;
    output->deleter = deleter;

    return output;
}

static inline void _setError(const char *what, char **error) {
    // size_t len = strlen(what);
    // *error = (char *)alloc(len * sizeof(char)+1);
    // strcpy(*error, what);
    // (*error)[len]='\0';
    *error = RedisModule_Strdup(what);
}

struct ModelContext {
    std::shared_ptr<tflite::FlatBufferModel> model;
    std::shared_ptr<tflite::Interpreter> interpreter;
    std::string buffer;
    DLDeviceType device;
    int64_t device_id;
};

} // namespace

extern "C" void tfliteBasicTest() {}

extern "C" void *tfliteLoadModel(const char *graph, size_t graphlen, DLDeviceType device,
                                 int64_t device_id, char **error) {
    std::string graphstr(graph, graphlen);

    std::shared_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    model = tflite::FlatBufferModel::BuildFromBuffer(graphstr.c_str(), graphlen);
    if (!model) {
        _setError("Failed to load model from buffer", error);
        return NULL;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter_);
    if (!interpreter_) {
        _setError("Failed to construct interpreter", error);
        return NULL;
    }

#if RAI_TFLITE_USE_CUDA
    if (device == DLDeviceType::kDLCUDA) {
        tflite::Interpreter::TfLiteDelegatePtr delegate =
            tflite::evaluation::CreateGPUDelegate(model.get());
        if (interpreter_->ModifyGraphWithDelegate(std::move(delegate)) != kTfLiteOk) {
            _setError("Failed to set GPU delegate", error);
            return NULL;
        }
    }
#endif

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        _setError("Failed to allocate tensors", error);
        return NULL;
    }

    std::shared_ptr<tflite::Interpreter> interpreter = std::move(interpreter_);

    ModelContext *ctx = new ModelContext();
    ctx->device = device;
    ctx->device_id = device_id;
    ctx->model = std::move(model);
    ctx->interpreter = std::move(interpreter);
    ctx->buffer = std::move(graphstr);

    return ctx;
}

extern "C" size_t tfliteModelNumInputs(void *ctx, char **error) {
    ModelContext *ctx_ = (ModelContext *)ctx;
    size_t ret = 0;
    try {
        auto interpreter = ctx_->interpreter;
        ret = interpreter->inputs().size();
    } catch (std::exception ex) {
        _setError(ex.what(), error);
    }
    return ret;
}

extern "C" const char *tfliteModelInputNameAtIndex(void *modelCtx, size_t index, char **error) {
    ModelContext *ctx_ = (ModelContext *)modelCtx;
    const char *ret = NULL;
    try {
        ret = ctx_->interpreter->GetInputName(index);
    } catch (std::exception ex) {
        _setError(ex.what(), error);
    }
    return ret;
}

extern "C" size_t tfliteModelNumOutputs(void *ctx, char **error) {
    ModelContext *ctx_ = (ModelContext *)ctx;
    size_t ret = 0;
    try {
        auto interpreter = ctx_->interpreter;
        ret = interpreter->outputs().size();
    } catch (std::exception ex) {
        _setError(ex.what(), error);
    }
    return ret;
}

extern "C" const char *tfliteModelOutputNameAtIndex(void *modelCtx, size_t index, char **error) {
    ModelContext *ctx_ = (ModelContext *)modelCtx;
    const char *ret = NULL;
    try {
        ret = ctx_->interpreter->GetOutputName(index);
    } catch (std::exception ex) {
        _setError(ex.what(), error);
    }
    return ret;
}

extern "C" void tfliteRunModel(void *ctx, long n_inputs, DLManagedTensor **inputs, long n_outputs,
                               DLManagedTensor **outputs, char **error) {
    ModelContext *ctx_ = (ModelContext *)ctx;

    auto interpreter = ctx_->interpreter;
    auto model = ctx_->model;

    const std::vector<int> tflite_inputs = interpreter->inputs();
    const std::vector<int> tflite_outputs = interpreter->outputs();

    if (n_inputs != tflite_inputs.size()) {
        _setError("Inconsistent number of inputs", error);
        return;
    }

    if (n_outputs != tflite_outputs.size()) {
        _setError("Inconsistent number of outputs", error);
        return;
    }

    // NOTE: TFLITE requires all tensors in the graph to be explicitly
    // preallocated before input tensors are memcopied. These are cached
    // in the session, so we need to check if for instance the batch size
    // has changed or the shape has changed in general compared to the
    // previous run and in that case we resize input tensors and call
    // the AllocateTensor function manually.
    bool need_reallocation = false;
    std::vector<int> dims;
    for (size_t i = 0; i < tflite_inputs.size(); i++) {
        const TfLiteTensor *tflite_tensor = interpreter->tensor(tflite_inputs[i]);
        int64_t ndim = inputs[i]->dl_tensor.ndim;
        int64_t *shape = inputs[i]->dl_tensor.shape;
        dims.resize(ndim);
        for (size_t j = 0; j < ndim; j++) {
            dims[j] = shape[j];
        }
        if (!tflite::EqualArrayAndTfLiteIntArray(tflite_tensor->dims, dims.size(), dims.data())) {
            if (interpreter->ResizeInputTensor(i, dims) != kTfLiteOk) {
                _setError("Failed to resize input tensors", error);
                return;
            }
            need_reallocation = true;
        }
    }

    if (need_reallocation) {
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            _setError("Failed to allocate tensors", error);
            return;
        }
    }

    try {
        for (size_t i = 0; i < tflite_inputs.size(); i++) {
            copyToTfLiteTensor(interpreter, tflite_inputs[i], inputs[i]);
        }
    } catch (std::exception &e) {
        _setError(e.what(), error);
        return;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        _setError("Failed to invoke TfLite", error);
        return;
    }

    try {
        for (size_t i = 0; i < tflite_outputs.size(); i++) {
            outputs[i] = toManagedDLPack(interpreter, tflite_outputs[i]);
        }
    } catch (std::exception &e) {
        _setError(e.what(), error);
        return;
    }
}

extern "C" void tfliteSerializeModel(void *ctx, char **buffer, size_t *len, char **error) {
    // NO OP
}

extern "C" void tfliteDeallocContext(void *ctx) {
    ModelContext *ctx_ = (ModelContext *)ctx;
    if (ctx_) {
        delete ctx_;
    }
}
