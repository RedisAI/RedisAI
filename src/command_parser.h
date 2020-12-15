#ifndef REDISAI_COMMAND_PARSER_H
#define REDISAI_COMMAND_PARSER_H

#include "redismodule.h"
#include "model.h"

enum RunCommands { CMD_MODELRUN = 0, CMD_SCRIPTRUN, CMD_DAGRUN };

/**
 * @brief  Validates MODELRUN command and write the model obtained from
 * the key space to the model pointer. The keys of the input and output tensord
 * are stored in the inkeys and outkeys arrays, the model key is saved in runkey,
 * and the given timeout is saved as well (if given, otherwise it is zero).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseModelRunCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, RAI_Model **model,
                         RAI_Error *error, RedisModuleString ***inkeys,
                         RedisModuleString ***outkets, RedisModuleString **runkey,
                         long long *timeout);

int ProcessRunCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, int command,
                      int dagMode);

#endif // REDISAI_COMMAND_PARSER_H
