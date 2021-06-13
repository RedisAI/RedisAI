#include <string>
#include <dlpack/dlpack.h>
#include "redisai.h"
#include "torch_redis.h"
#include "../torch_c.h"

torch::IValue IValueFromRedisReply(RedisModuleCallReply *reply){

    int reply_type = RedisModule_CallReplyType(reply);
    switch(reply_type) {
        case REDISMODULE_REPLY_NULL: {
            return torch::IValue();
        }
        case REDISMODULE_REPLY_STRING: {
            size_t len;
            const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
            std::string str = replyStr;
            return torch::IValue(str.substr(0,len));
        }
        case REDISMODULE_REPLY_INTEGER: {
            int intValue = (int)RedisModule_CallReplyInteger(reply);
            return torch::IValue(intValue);
        }
        case REDISMODULE_REPLY_ARRAY: {
            c10::impl::GenericList vec = c10::impl::GenericList(c10::AnyType::create());
            size_t len = RedisModule_CallReplyLength(reply);
            for (auto i = 0; i < len; ++i) {
                RedisModuleCallReply *subReply = RedisModule_CallReplyArrayElement(reply, i);
                torch::IValue value = IValueFromRedisReply(subReply);
                vec.push_back(value);
            }
            return torch::IValue(vec);
        }
        case REDISMODULE_REPLY_ERROR: {
            size_t len;
            const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
            throw std::runtime_error(replyStr);
        }
        default:{
            throw(std::runtime_error("Unsupported redis type"));
        }
    }
}

torch::IValue redisExecute(const std::string& fn_name, const std::vector<std::string> &args ) {
    RedisModuleCtx* ctx = RedisModule_GetThreadSafeContext(nullptr);
    RedisModule_ThreadSafeContextLock(ctx);
    size_t len = args.size();
    RedisModuleString* arguments[len];
    len = 0;
    for (auto &arg : args) {
        const char* str = arg.c_str();
        arguments[len++] = RedisModule_CreateString(ctx, str, strlen(str));
    }

    RedisModuleCallReply *reply = RedisModule_Call(ctx, fn_name.c_str(), "!v", arguments, len);
    RedisModule_ThreadSafeContextUnlock(ctx);
    torch::IValue value = IValueFromRedisReply(reply);
    RedisModule_FreeCallReply(reply);
    RedisModule_FreeThreadSafeContext(ctx);
    for(int i= 0; i < len; i++){
        RedisModule_FreeString(nullptr, arguments[i]);
    }
    return value;
}

torch::List<torch::IValue> asList(const torch::IValue &v) {
    return v.toList();
}

std::vector<torch::Tensor> modelExecute(const std::string& model_key, const std::vector<torch::Tensor> &inputs) {
    RedisModuleCtx* ctx = RedisModule_GetThreadSafeContext(nullptr);
    RedisModule_ThreadSafeContextLock(ctx);

    // Prepare for getting model from key space.
    const char* model_key_str = model_key.c_str();
    RedisModuleString *model_key_rs = RedisModule_CreateString(ctx, model_key_str,
      strlen(model_key_str));
    RAI_Error *err;
    RedisAI_InitError(&err);
    RAI_Model *model = nullptr;
    RAI_ModelRunCtx *model_run_ctx = nullptr;
    std::vector<torch::Tensor> outputs;

    int status = RedisAI_GetModelFromKeyspace(ctx, model_key_rs, &model,
      REDISMODULE_READ, err);
    RedisModule_FreeString(nullptr, model_key_rs);
    if (status != REDISMODULE_OK) {
        goto finish;
    }

    // Create model run ctx and store the input tensors in it.
    model_run_ctx = RedisAI_ModelRunCtxCreate(model);
    for (auto &input : inputs) {
        DLManagedTensor *dl_tensor = torchTensorPtrToManagedDLPack(&input);
        RAI_Tensor *tensor = RedisAI_TensorCreateFromDLTensor(dl_tensor);
        RedisAI_ModelRunCtxAddInput(model_run_ctx, nullptr, tensor);
    }

    // Run the model, if finished successfully, load the outputs.
    status = RedisAI_ModelRun(&model_run_ctx, 1, err);
    if (status != REDISMODULE_OK) {
        goto finish;
    }
    outputs = std::vector<torch::Tensor>(RedisAI_ModelRunCtxNumOutputs(model_run_ctx));
    for (size_t i = 0; i < RedisAI_ModelRunCtxNumOutputs(model_run_ctx); i++) {
        RAI_Tensor *tensor = RedisAI_ModelRunCtxOutputTensor(model_run_ctx, i);
        DLTensor *dl_tensor = RedisAI_TensorGetDLTensor(tensor);
        auto *output = static_cast<torch::Tensor *>(torchTensorPtrFromDLPack(dl_tensor));
        outputs.push_back(*output);
        delete output;
    }

    finish:
    RedisModule_ThreadSafeContextUnlock(ctx);
    RedisModule_FreeThreadSafeContext(ctx);
    if (model_run_ctx) {
        RedisAI_ModelRunCtxFree(model_run_ctx);
    }
    if (status != REDISMODULE_OK) {
        const char* error = RedisAI_GetError(err);
        RedisAI_FreeError(err);
        throw std::runtime_error(error);
    }
    return outputs;
}