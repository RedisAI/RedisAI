#include "execution_ctx.h"
#include "redismodule.h"
#include "util/arr.h"

RAI_ExecutionCtx* RAI_ExecutionCtx_New(RAI_ExecutionCtx_Free_fn freeFn) {
    RAI_ExecutionCtx* ctx = RedisModule_Alloc(sizeof(RAI_ExecutionCtx));
    ctx->inputs = array_new(RAI_Tensor*, 10);
    ctx->outputs = array_new(RAI_Tensor*, 10);
    ctx->freeFn = freeFn;
    return ctx;
}
void RAI_ExecutionCtx_Free(RAI_ExecutionCtx* ctx) {
    size_t inputsLen = array_len(ctx->inputs);
    for(size_t i = 0; i < inputsLen; i++) {
        RAI_TensorFree(ctx->inputs[i]);
    }
    size_t outputsLen = array_len(ctx->outputs);
    for(size_t i = 0; i < outputsLen; i++) {
        RAI_TensorFree(ctx->outputs[i]);
    }
    array_free(ctx->inputs);
    array_free(ctx->outputs);
    RedisModule_Free(ctx);
}

inline size_t RAI_ExecutionCtx_InputsLen(RAI_ExecutionCtx* ctx) {
    return array_len(ctx->inputs);
}
inline void RAI_ExecutionCtx_AddInput(RAI_ExecutionCtx* ctx, RAI_Tensor* t) {
    ctx->inputs = array_append(ctx->inputs, t);
}

inline RAI_Tensor* RAI_ExecutionCtx_GetInput(RAI_ExecutionCtx* ctx, size_t index) {
    RedisModule_Assert(index< array_len(ctx->inputs));
    return ctx->inputs[index];
}

inline size_t RAI_ExecutionCtx_OutputsLen(RAI_ExecutionCtx* ctx) {
    return array_len(ctx->outputs);
}

inline void RAI_ExecutionCtx_AddOuputPlaceholder(RAI_ExecutionCtx* ctx) {
    ctx->outputs = array_append(ctx->outputs, NULL);
}

inline RAI_Tensor* RAI_ExecutionCtx_GetOutput(RAI_ExecutionCtx* ctx, size_t index) {
    RedisModule_Assert(index < array_len(ctx->outputs));
    return ctx->outputs[index];
}
