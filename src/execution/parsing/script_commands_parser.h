/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "redismodule.h"
#include "execution/run_info.h"

/**
 * @brief  Parse and validate SCRIPTEXECUTE command: create a scriptRunCtx based on the script
 * obtained from the key space and the function name given, and save it in the op. The keys of the
 * input and output tensors are stored in the op's inkeys and outkeys arrays,
 * and the given timeout is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseScriptExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp,
                              RedisModuleString **argv, int argc);
