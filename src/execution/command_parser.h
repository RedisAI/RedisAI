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
 * @brief  Parse and execute RedisAI run command. After parsing and validation, the resulted
 * runInfo (DAG) is queued and the client is blocked until the execution is complete (async
 * execution).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int RedisAI_ExecuteCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                           RunCommand command, bool ro_dag);
