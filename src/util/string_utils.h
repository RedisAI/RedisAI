#include "redismodule.h"
#include "dict.h"

RedisModuleString *RAI_HoldString(RedisModuleCtx *ctx, RedisModuleString *str);

uint64_t RAI_StringsHashFunction(const void *key);
int RAI_StringsKeyCompare(void *privdata, const void *key1, const void *key2);
void RAI_StringsKeyDestructor(void *privdata, void *key);
void *RAI_StringsKeyDup(void *privdata, const void *key);

uint64_t RAI_RStringsHashFunction(const void *key);
int RAI_RStringsKeyCompare(void *privdata, const void *key1, const void *key2);
void RAI_RStringsKeyDestructor(void *privdata, void *key);
void *RAI_RStringsKeyDup(void *privdata, const void *key);

