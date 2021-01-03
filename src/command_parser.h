#pragma once

#include "redismodule.h"
#include "run_info.h"

typedef enum RunCommand { CMD_MODELRUN = 0, CMD_SCRIPTRUN, CMD_DAGRUN } RunCommand;

/**
 * @brief  Parse and validate MODELRUN command: create a modelRunCtx based on the model obtained
 * from the key space and save it in the op. The keys of the input and output tensors are stored in
 * the op's inkeys and outkeys arrays, the model key is saved in op's runkey, and the given timeout
 * is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseModelRunCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleCtx *ctx,
                         RedisModuleString **argv, int argc);

/**
 * @brief  Parse and validate SCRIPTRUN command: create a scriptRunCtx based on the script obtained
 * from the key space and the function name given, and save it in the op. The keys of the input and
 * output tensors are stored in the op's inkeys and outkeys arrays, the script key is saved in op's
 * runkey, and the given timeout is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseScriptRunCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleCtx *ctx,
                          RedisModuleString **argv, int argc);

/**
 * @brief  Parse and execute RedisAI run command. After parsing and validation, the resulted
 * runInfo (DAG) is queued and the client is blocked until the execution is complete (async
 * execution).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int RedisAI_ExecuteCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                           RunCommand command, bool ro_dag);
