#pragma once

#include "redismodule.h"
#include "run_info.h"

typedef enum RunCommand {
    CMD_MODELRUN = 0,
    CMD_SCRIPTRUN,
    CMD_DAGRUN,
    CMD_MODELEXECUTE,
    CMD_SCRIPTEXECUTE
} RunCommand;

/**
 * @brief  Parse and validate MODELEXECUTE command: create a modelRunCtx based on the model obtained
 * from the key space and save it in the op. The keys of the input and output tensors are stored in
 * the op's inkeys and outkeys arrays, the model key is saved in op's runkey, and the given timeout
 * is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseModelExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                             int argc);


/**
 * @brief  Parse and validate SCRIPTEXECUTE command: create a scriptRunCtx based on the script obtained
 * from the key space and the function name given, and save it in the op. The keys of the input and
 * output tensors are stored in the op's inkeys and outkeys arrays, the script key is saved in op's
 * runkey, and the given timeout is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseModelExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                             int argc);
/**
 * Extract the params for the ModelCtxRun object from AI.MODELEXECUTE arguments.
 *
 * @param ctx Context in which Redis modules operate
 * @param inkeys Model input tensors keys, as an array of strings
 * @param outkeys Model output tensors keys, as an array of strings
 * @param mctx Destination Model context to store the parsed data
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise
 */

int ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                          RedisModuleString **outkeys, RAI_ModelRunCtx *mctx, RAI_Error *err);

/**
 * @brief  Parse and validate TIMEOUT argument. If it is valid, store it in timeout.
 * Otherwise set an error.
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout);

/**
 * @brief  Parse and execute RedisAI run command. After parsing and validation, the resulted
 * runInfo (DAG) is queued and the client is blocked until the execution is complete (async
 * execution).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int RedisAI_ExecuteCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                           RunCommand command, bool ro_dag);
