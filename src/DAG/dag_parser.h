#pragma once

#include "run_info.h"

int DAG_CommandParser(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, bool dag_ro,
                      RedisAI_RunInfo **rinfo_ptr);
