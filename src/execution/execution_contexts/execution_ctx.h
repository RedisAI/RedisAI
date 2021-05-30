#pragma once

#include "redis_ai_objects/tensor.h"

// Pre decleration
typedef struct RAI_ExecutionCtx RAI_ExecutionCtx;

typedef void (*RAI_ExecutionCtx_Free_fn)(RAI_ExecutionCtx *ctx);

typedef struct RAI_ExecutionCtx {
    RAI_Tensor **inputs;
    RAI_Tensor **outputs;
    RAI_ExecutionCtx_Free_fn freeFn;
} RAI_ExecutionCtx;

void RAI_ExecutionCtx_Init(RAI_ExecutionCtx *ctx, RAI_ExecutionCtx_Free_fn freeFn);
void RAI_ExecutionCtx_Free(RAI_ExecutionCtx *ctx);

size_t RAI_ExecutionCtx_NumInputs(RAI_ExecutionCtx *ctx);
void RAI_ExecutionCtx_AddInput(RAI_ExecutionCtx *ctx, RAI_Tensor *t);
RAI_Tensor *RAI_ExecutionCtx_GetInput(RAI_ExecutionCtx *ctx, size_t index);

size_t RAI_ExecutionCtx_NumOutputs(RAI_ExecutionCtx *ctx);
void RAI_ExecutionCtx_AddOuputPlaceholder(RAI_ExecutionCtx *ctx);
void RAI_ExecutionCtx_SetOutput(RAI_ExecutionCtx *ctx, RAI_Tensor *t, size_t index);
RAI_Tensor *RAI_ExecutionCtx_GetOutput(RAI_ExecutionCtx *ctx, size_t index);