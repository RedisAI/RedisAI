
#pragma once

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
