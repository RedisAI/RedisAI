#pragma once
#include "redismodule.h"
#include "execution/run_info.h"

/**
 * @brief  Parse and validate SCRIPTEXECUTE command: create a scriptRunCtx based on the script
 * obtained from the key space and the function name given, and save it in the op. The keys of the
 * input and output tensors are stored in the op's inkeys and outkeys arrays, the script key is
 * saved in op's runkey, and the given timeout is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseScriptExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp,
                              RedisModuleString **argv, int argc);

// /**
//  * @brief  Parse and validate SCRIPTEXECUTE from within a DAGEXECUTEE command. This function
//  parse
//  * the script scopes INPUTS, LIST_INPOUTS, OUTPUTS, without the KEYS scope as it will begiven in
//  the
//  * DAG command.
//  * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
//  */
// int ParseScriptExecuteArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
//                                            RAI_Error *error, RedisModuleString ***inkeys,
//                                            RedisModuleString ***outkeys, RAI_ScriptRunCtx *sctx,
//                                            long long *timeout);