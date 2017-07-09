#include "redismodule.h"
#include "tensorflow/c/c_api.h"

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc){
  if (RedisModule_Init(ctx, "tf", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  return REDISMODULE_OK;
}
