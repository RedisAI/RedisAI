
#ifndef TEST_UNIT_RMALLOC_H_
#define TEST_UNIT_RMALLOC_H_

#include "../../src/redismodule.h"
#include <stdlib.h>
#include <string.h>

void Alloc_Reset() {
    RedisModule_Alloc = malloc;
    RedisModule_Realloc = realloc;
    RedisModule_Calloc = calloc;
    RedisModule_Free = free;
    RedisModule_Strdup = strdup;
}

#endif /* TEST_UNIT_RMALLOC_H_ */
