/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "redismodule.h"
#include "dict.h"

RedisModuleString *RAI_HoldString(RedisModuleString *str);
void RAI_StringToUpper(const char *str, char *upper, size_t str_len);

uint64_t RAI_StringsHashFunction(const void *key);
int RAI_StringsKeyCompare(void *privdata, const void *key1, const void *key2);
void RAI_StringsKeyDestructor(void *privdata, void *key);
void *RAI_StringsKeyDup(void *privdata, const void *key);

uint64_t RAI_RStringsHashFunction(const void *key);
int RAI_RStringsKeyCompare(void *privdata, const void *key1, const void *key2);
void RAI_RStringsKeyDestructor(void *privdata, void *key);
void *RAI_RStringsKeyDup(void *privdata, const void *key);
