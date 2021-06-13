#pragma once

#include <stdint.h>

uintptr_t (*RedisAI_GetThreadId)(void);

uintptr_t (*RedisAI_GetThreadsCount)(void);

long long (*RedisAI_GetNumThreadsPerQueue)(void);

long long (*RedisAI_GetModelExecutionTimeout)(void);
