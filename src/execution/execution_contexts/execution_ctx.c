#include "execution_ctx.h"
#include "redismodule.h"
#include "util/arr.h"

void RAI_ExecutionCtx_Init(RAI_ExecutionCtx *ctx, RAI_RunStats *run_stats,
                           RAI_ExecutionCtx_Free_fn freeFn) {
    ctx->inputs = array_new(RAI_Tensor *, 10);
    ctx->outputs = array_new(RAI_Tensor *, 10);
    ctx->runStats = RAI_StatsGetShallowCopy(run_stats);
    ctx->freeFn = freeFn;
}
void RAI_ExecutionCtx_Free(RAI_ExecutionCtx *ctx) {
    size_t inputsLen = array_len(ctx->inputs);
    for (size_t i = 0; i < inputsLen; i++) {
        RAI_TensorFree(ctx->inputs[i]);
    }
    size_t outputsLen = array_len(ctx->outputs);
    for (size_t i = 0; i < outputsLen; i++) {
        RAI_TensorFree(ctx->outputs[i]);
    }
    array_free(ctx->inputs);
    array_free(ctx->outputs);
    // In case that the model/script was deleted while the execution took place in the background,
    // and this is the last execution of this model - release the run stats structure.
    RAI_StatsFree(ctx->runStats);
}

inline size_t RAI_ExecutionCtx_NumInputs(RAI_ExecutionCtx *ctx) { return array_len(ctx->inputs); }

inline void RAI_ExecutionCtx_AddInput(RAI_ExecutionCtx *ctx, RAI_Tensor *t) {
    if (t != NULL) {
        t = RAI_TensorGetShallowCopy(t);
    }
    ctx->inputs = array_append(ctx->inputs, t);
}

inline RAI_Tensor *RAI_ExecutionCtx_GetInput(RAI_ExecutionCtx *ctx, size_t index) {
    RedisModule_Assert(index < array_len(ctx->inputs));
    return ctx->inputs[index];
}

inline size_t RAI_ExecutionCtx_NumOutputs(RAI_ExecutionCtx *ctx) { return array_len(ctx->outputs); }

inline void RAI_ExecutionCtx_AddOutputPlaceholder(RAI_ExecutionCtx *ctx) {
    ctx->outputs = array_append(ctx->outputs, NULL);
}

void RAI_ExecutionCtx_SetOutput(RAI_ExecutionCtx *ctx, RAI_Tensor *t, size_t index) {
    RedisModule_Assert(index < array_len(ctx->outputs));
    ctx->outputs[index] = t;
}

inline RAI_Tensor *RAI_ExecutionCtx_GetOutput(RAI_ExecutionCtx *ctx, size_t index) {
    RedisModule_Assert(index < array_len(ctx->outputs));
    return ctx->outputs[index];
}

inline RAI_RunStats *RAI_ExecutionCtx_GetStats(RAI_ExecutionCtx *ctx) { return ctx->runStats; }
