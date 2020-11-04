#include "encode_v100.h"

void RAI_RDBSaveTensor_v100(RedisModuleIO *io, void *value) {
    RAI_Tensor *tensor = (RAI_Tensor*)value;

    size_t ndim = tensor->tensor.dl_tensor.ndim;

    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.ctx.device_type);
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.ctx.device_id);
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.bits);
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.code);
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.lanes);
    RedisModule_SaveUnsigned(io, ndim);
    for (size_t i=0; i<ndim; i++) {
        RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.shape[i]);
    }
    for (size_t i=0; i<ndim; i++) {
        RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.strides[i]);
    }
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.byte_offset);
    size_t size = RAI_TensorByteSize(tensor);

    RedisModule_SaveStringBuffer(io, tensor->tensor.dl_tensor.data, size);
}
