/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <redis_ai_objects/stats.h>
#include "redis_ai_objects/tensor.h"

// Pre decleration
typedef struct RAI_ExecutionCtx RAI_ExecutionCtx;

typedef void (*RAI_ExecutionCtx_Free_fn)(RAI_ExecutionCtx *ctx);

/**
 * @brief Generic struct to hold execution contexts for DAG ops. This struct holds the input and
 * output tensors of the op, as well as inheriting classes specific functionality.
 *
 */
typedef struct RAI_ExecutionCtx {
    RAI_Tensor **inputs;             // DAG op input tensors.
    RAI_Tensor **outputs;            // DAG op output tensors.
    RAI_RunStats *runStats;          // The underline op's (Model/Script) stats entry.
    RAI_ExecutionCtx_Free_fn freeFn; // Inheriting execution context free function.
} RAI_ExecutionCtx;

/**
 * @brief Initializes an allocated RAI_ExecutionCtx.
 *
 * @param ctx - Execution context to initialize.
 * @param freeFn - Specific free function for inheriting execution contexts (script or model)
 */
void RAI_ExecutionCtx_Init(RAI_ExecutionCtx *ctx, RAI_RunStats *run_stats,
                           RAI_ExecutionCtx_Free_fn freeFn);

/**
 * @brief Frees the execution context internal structures. To be used from an inhereting execution
 * contxt.
 *
 * @param ctx - Execution context to Free.
 */
void RAI_ExecutionCtx_Free(RAI_ExecutionCtx *ctx);

/**
 * @brief Returns the number of input tensors of the execution context.
 *
 * @param ctx - Execution context.
 * @return size_t - Number of input tensors.
 */
size_t RAI_ExecutionCtx_NumInputs(RAI_ExecutionCtx *ctx);

/**
 * @brief Adds an input tensor to the execution context.
 *
 * @param ctx - Execution context.
 * @param t - Input tensor.
 */
void RAI_ExecutionCtx_AddInput(RAI_ExecutionCtx *ctx, RAI_Tensor *t);

/**
 * @brief Returns an input tensor from the execution context, for a given index.
 *
 * @param ctx - Execution context.
 * @param index
 * @return RAI_Tensor* - Input tensor.
 */
RAI_Tensor *RAI_ExecutionCtx_GetInput(RAI_ExecutionCtx *ctx, size_t index);

/**
 * @brief Returns the number of output tensors/placeholders of the execution context.
 *
 * @param ctx - Execution context.
 * @return size_t - Number of output tensors/placeholders.
 */
size_t RAI_ExecutionCtx_NumOutputs(RAI_ExecutionCtx *ctx);

/**
 * @brief Sets (appends) an output tensor placeholder to the execution context.
 *
 * @param ctx - Execution context.
 */
void RAI_ExecutionCtx_AddOutputPlaceholder(RAI_ExecutionCtx *ctx);

/**
 * @brief Sets an output tensor in a specfic index, populated before by a placeholder.
 *
 * @param ctx - Execution context.
 * @param t - Output tensor.
 * @param index
 */
void RAI_ExecutionCtx_SetOutput(RAI_ExecutionCtx *ctx, RAI_Tensor *t, size_t index);

/**
 * @brief Returns an output tensor from the execution context, for a given index.
 *
 * @param ctx - Execution context.
 * @param index
 * @return RAI_Tensor* - Output tensor.
 */
RAI_Tensor *RAI_ExecutionCtx_GetOutput(RAI_ExecutionCtx *ctx, size_t index);

/**
 * @brief Returns the RunStats object for underline object.
 * @param ctx - Execution context.
 * @return RAI_RunStats
 */
RAI_RunStats *RAI_ExecutionCtx_GetStats(RAI_ExecutionCtx *ctx);
