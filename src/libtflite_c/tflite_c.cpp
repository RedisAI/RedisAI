#include "tflite_c.h"
#include <iostream>
#include <sstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
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
    default:
        break;
    }
    return dtype;
}

static DLContext getDLContext(const TfLiteTensor *tensor, const int64_t &device_id) {
    DLContext ctx;
    ctx.device_id = device_id;
    // if (tensor->.is_cuda()) {
    //   ctx.device_type = DLDeviceType::kDLGPU;
    // } else {
    //   ctx.device_type = DLDeviceType::kDLCPU;
    // }
    ctx.device_type = DLDeviceType::kDLCPU;
    return ctx;
}

#if 0
static at::DeviceType getATenDeviceType(DLDeviceType device_type) {
  switch (device_type) {
    case DLDeviceType::kDLCPU:
      return at::DeviceType::CPU;
    case DLDeviceType::kDLGPU:
      return at::DeviceType::CUDA;
    case DLDeviceType::kDLOpenCL:
      return at::DeviceType::OPENCL;
    case DLDeviceType::kDLROCM:
      return at::DeviceType::HIP;
    default:
      throw std::logic_error("Unsupported device_type: " + std::to_string(device_type));
  }
  return at::DeviceType::CPU; // impossible
}
#endif

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

    switch (tensor->type) {
    case kTfLiteUInt8:
        memcpy(interpreter->typed_tensor<uint8_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt64:
        memcpy(interpreter->typed_tensor<int64_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt32:
        memcpy(interpreter->typed_tensor<int32_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt16:
        memcpy(interpreter->typed_tensor<int16_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteInt8:
        memcpy(interpreter->typed_tensor<int8_t>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteFloat32:
        memcpy(interpreter->typed_tensor<float>(tflite_input), input->dl_tensor.data, nbytes);
        break;
    case kTfLiteFloat16:
        throw std::logic_error("Float16 not currently supported as input tensor data type");
        break;
    default:
        throw std::logic_error("Unsupported input data type");
    }
}

void deleter(DLManagedTensor *arg) {
    delete[](uint8_t *) arg->dl_tensor.data;
    delete[] arg->dl_tensor.shape;
    delete[] arg->dl_tensor.strides;
    // FIXME
    // delete arg;
}

DLManagedTensor *toManagedDLPack(std::shared_ptr<tflite::Interpreter> interpreter,
                                 int tflite_output, void *(*alloc)(size_t)) {
    TfLiteTensor *tensor = interpreter->tensor(tflite_output);

    TfLiteIntArray *output_dims = tensor->dims;

    DLDataType dtype = getDLDataType(tensor);

    int64_t device_id = 0;
    DLContext ctx = getDLContext(tensor, device_id);

    DLTensor dl_tensor = (DLTensor){.data = new uint8_t[tensor->bytes],
                                    .ctx = ctx,
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
    case kTfLiteFloat16:
        throw std::logic_error("Float16 not currently supported as output tensor data type");
        break;
    default:
        throw std::logic_error("Unsupported output data type");
    }

    // We use alloc here to allow deallocation from the module
    DLManagedTensor *output = (DLManagedTensor *)alloc(sizeof(DLManagedTensor));
    output->dl_tensor = dl_tensor;
    output->manager_ctx = NULL;
    output->deleter = deleter;

    return output;
}

void setError(const char *what, char **error, void *(*alloc)(size_t)) {
    size_t len = strlen(what);
    *error = (char *)alloc(len * sizeof(char));
    strcpy(*error, what);
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
                                 int64_t device_id, char **error, void *(*alloc)(size_t)) {
    std::string graphstr(graph, graphlen);

    std::shared_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    model = tflite::FlatBufferModel::BuildFromBuffer(graphstr.c_str(), graphlen);
    if (!model) {
        setError("Failed to load model from buffer", error, alloc);
        return NULL;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter_);
    if (!interpreter_) {
        setError("Failed to construct interpreter", error, alloc);
        return NULL;
    }

#if RAI_TFLITE_USE_CUDA
    if (device == DLDeviceType::kDLGPU) {
        tflite::Interpreter::TfLiteDelegatePtr delegate =
            tflite::evaluation::CreateGPUDelegate(model.get());
        if (interpreter_->ModifyGraphWithDelegate(std::move(delegate)) != kTfLiteOk) {
            setError("Failed to set GPU delegate", error, alloc);
            return NULL;
        }
    }
#endif

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        setError("Failed to allocate tensors", error, alloc);
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

extern "C" void tfliteRunModel(void *ctx, long n_inputs, DLManagedTensor **inputs, long n_outputs,
                               DLManagedTensor **outputs, char **error, void *(*alloc)(size_t)) {
    ModelContext *ctx_ = (ModelContext *)ctx;

    auto interpreter = ctx_->interpreter;
    auto model = ctx_->model;

    const std::vector<int> tflite_inputs = interpreter->inputs();
    const std::vector<int> tflite_outputs = interpreter->outputs();

    if (n_inputs != tflite_inputs.size()) {
        setError("Inconsistent number of inputs", error, alloc);
        return;
    }

    if (n_outputs != tflite_outputs.size()) {
        setError("Inconsistent number of outputs", error, alloc);
        return;
    }

    try {
        for (size_t i = 0; i < tflite_inputs.size(); i++) {
            copyToTfLiteTensor(interpreter, tflite_inputs[i], inputs[i]);
        }
    } catch (std::exception &e) {
        setError(e.what(), error, alloc);
        return;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        setError("Failed to invoke TfLite", error, alloc);
        return;
    }

    try {
        for (size_t i = 0; i < tflite_outputs.size(); i++) {
            outputs[i] = toManagedDLPack(interpreter, tflite_outputs[i], alloc);
        }
    } catch (std::exception &e) {
        setError(e.what(), error, alloc);
        return;
    }
}

extern "C" void tfliteSerializeModel(void *ctx, char **buffer, size_t *len, char **error,
                                     void *(*alloc)(size_t)) {
    // NO OP
}

extern "C" void tfliteDeallocContext(void *ctx) {
    ModelContext *ctx_ = (ModelContext *)ctx;
    if (ctx_) {
        delete ctx_;
    }
}
