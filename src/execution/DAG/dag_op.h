/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "redismodule.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/script.h"
#include "redis_ai_objects/model_struct.h"
#include "execution/execution_contexts/execution_ctx.h"

typedef enum DAGCommand {
    REDISAI_DAG_CMD_NONE = 0,
    REDISAI_DAG_CMD_TENSORSET,
    REDISAI_DAG_CMD_TENSORGET,
    REDISAI_DAG_CMD_MODELRUN,
    REDISAI_DAG_CMD_SCRIPTRUN
} DAGCommand;

#define VALIDATE_DAG_COMMAND(cmd)                                                                  \
    RedisModule_Assert(cmd >= REDISAI_DAG_CMD_TENSORSET && cmd <= REDISAI_DAG_CMD_SCRIPTRUN);

typedef struct RAI_DagOp {
    DAGCommand commandType;
    RedisModuleString **inkeys;
    RedisModuleString **outkeys;
    size_t *inkeys_indices;
    size_t *outkeys_indices;
    RAI_Tensor *outTensor; // The tensor to upload in TENSORSET op.
    RAI_ExecutionCtx *ectx;
    uint fmt; // This is relevant for TENSORGET op.
    char *devicestr;
    int result; // REDISMODULE_OK or REDISMODULE_ERR
    long long duration_us;
    RAI_Error *err;
    RedisModuleString **argv;
    int argc;
} RAI_DagOp;

/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitDagOp(RAI_DagOp **result);

/**
 * Frees the memory allocated of RAI_DagOp
 * @param ctx Context in which Redis modules operate
 * @param RAI_DagOp context in which RedisAI command operates.
 */
void RAI_FreeDagOp(RAI_DagOp *dagOp);
