#pragma once

#include "run_info.h"

int DAG_CommandParser(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, int dagMode,
                      RedisAI_RunInfo **rinfo_ptr);

#endif // REDISAI_DAG_PARSER_H
