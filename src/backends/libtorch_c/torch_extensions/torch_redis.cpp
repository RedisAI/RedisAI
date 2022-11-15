/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#define BACKENDS_API_EXTERN
#include <string>
#include <dlpack/dlpack.h>
#include "backends/backends_api.h"
#include "torch_redis.h"
#include "../torch_c.h"

static torch::RegisterOperators registry;
void registerRedisOps(void) {
    registry = torch::RegisterOperators("redis::execute", &redisExecute)
                   .op("redis::asList", &asList)
                   .op("redisAI::model_execute", &modelExecute);
}

torch::IValue IValueFromRedisReply(RedisModuleCtx *ctx, RedisModuleCallReply *reply) {

    int reply_type = RedisModule_CallReplyType(reply);
    switch (reply_type) {
    case REDISMODULE_REPLY_NULL: {
        return torch::IValue();
    }
    case REDISMODULE_REPLY_STRING: {
        size_t len;
        const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
        std::string str = replyStr;
        return torch::IValue(str.substr(0, len));
    }
    case REDISMODULE_REPLY_INTEGER: {
        int intValue = (int)RedisModule_CallReplyInteger(reply);
        return torch::IValue(intValue);
    }
    case REDISMODULE_REPLY_ARRAY: {
        c10::List<c10::IValue> vec(c10::AnyType::get());
        size_t len = RedisModule_CallReplyLength(reply);
        for (auto i = 0; i < len; ++i) {
            RedisModuleCallReply *subReply = RedisModule_CallReplyArrayElement(reply, i);
            torch::IValue value = IValueFromRedisReply(ctx, subReply);
            vec.push_back(value);
        }
        return torch::IValue(vec);
    }
    case REDISMODULE_REPLY_ERROR: {
        size_t len;
        const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
        std::string error_str = "Redis command returned an error: " + std::string(replyStr);
        RedisModule_FreeCallReply(reply);
        RedisModule_FreeThreadSafeContext(ctx);
        throw std::runtime_error(error_str);
    }
    case REDISMODULE_REPLY_UNKNOWN: {
        std::string error_str = "Redis command returned an error: " + std::string(strerror(errno));
        RedisModule_FreeThreadSafeContext(ctx);
        throw(std::runtime_error(error_str));
    }
    default: {
        RedisModule_FreeThreadSafeContext(ctx);
        throw(std::runtime_error("Unexpected internal error"));
    }
    }
}

torch::IValue redisExecute(const std::string &fn_name, const std::vector<std::string> &args) {
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(nullptr);
    RedisModule_ThreadSafeContextLock(ctx);
    size_t len = args.size();
    RedisModuleString *arguments[len];
    len = 0;
    for (auto &arg : args) {
        const char *str = arg.c_str();
        arguments[len++] = RedisModule_CreateString(ctx, str, strlen(str));
    }
    RedisModuleCallReply *reply = RedisModule_Call(ctx, fn_name.c_str(), "!v", arguments, len);
    RedisModule_ThreadSafeContextUnlock(ctx);
    for (int i = 0; i < len; i++) {
        RedisModule_FreeString(nullptr, arguments[i]);
    }
    torch::IValue value = IValueFromRedisReply(ctx, reply);
    RedisModule_FreeCallReply(reply);
    RedisModule_FreeThreadSafeContext(ctx);
    return value;
}

torch::List<torch::IValue> asList(const torch::IValue &v) { return v.toList(); }

std::vector<torch::Tensor> modelExecute(const std::string &model_key,
                                        const std::vector<torch::Tensor> &inputs,
                                        int64_t num_outputs) {
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(nullptr);

    // Prepare for getting model from key space.
    const char *model_key_str = model_key.c_str();
    RedisModuleString *model_key_rs =
        RedisModule_CreateString(ctx, model_key_str, model_key.size());
    RAI_Error *err;
    RedisAI_InitError(&err);
    RAI_Model *model = nullptr;
    RAI_ModelRunCtx *model_run_ctx = nullptr;
    std::vector<torch::Tensor> outputs;

    RedisModule_ThreadSafeContextLock(ctx);
    int status = RedisAI_GetModelFromKeyspace(ctx, model_key_rs, &model, REDISMODULE_READ, err);
    RedisModule_FreeString(nullptr, model_key_rs);
    if (status != REDISMODULE_OK) {
        RedisModule_ThreadSafeContextUnlock(ctx);
        RedisModule_FreeThreadSafeContext(ctx);
        goto finish;
    }

    // Create model run ctx, store the input tensors and output placeholders in it.
    model_run_ctx = RedisAI_ModelRunCtxCreate(model);
    RedisModule_ThreadSafeContextUnlock(ctx);
    RedisModule_FreeThreadSafeContext(ctx);
    for (auto &input : inputs) {
        DLManagedTensor *dl_tensor = torchTensorPtrToManagedDLPack(&input);
        RAI_Tensor *tensor = RedisAI_TensorCreateFromDLTensor(dl_tensor);
        RedisAI_ModelRunCtxAddInput(model_run_ctx, nullptr, tensor);
        // Decrease ref count, ownership belongs to model_run_ctx now.
        RedisAI_TensorFree(tensor);
    }
    for (int i = 0; i < num_outputs; i++) {
        RedisAI_ModelRunCtxAddOutput(model_run_ctx, nullptr);
    }

    // Run the model, if finished successfully, load the outputs.
    status = RedisAI_ModelRun(&model_run_ctx, 1, err);
    if (status != REDISMODULE_OK) {
        goto finish;
    }
    for (size_t i = 0; i < RedisAI_ModelRunCtxNumOutputs(model_run_ctx); i++) {
        RAI_Tensor *tensor = RedisAI_ModelRunCtxOutputTensor(model_run_ctx, i);
        RedisAI_TensorGetShallowCopy(tensor);
        torch::Tensor output;
        torchTensorFromRAITensor(tensor, static_cast<void *>(&output));
        outputs.push_back(output);
    }

finish:
    if (model_run_ctx) {
        RedisAI_ModelRunCtxFree(model_run_ctx);
    }
    if (status != REDISMODULE_OK) {
        auto error = std::make_shared<std::string>(RedisAI_GetError(err));
        RedisAI_FreeError(err);
        throw std::runtime_error(*error);
    }
    RedisAI_FreeError(err);
    return outputs;
}
