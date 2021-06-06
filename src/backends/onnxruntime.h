#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model.h"
#include "execution/execution_contexts/execution_ctx.h"

unsigned long long RAI_GetMemoryInfoORT(void);

unsigned long long RAI_GetMemoryAccessORT(void);

pthread_key_t (*RedisAI_ThreadId)(void);
long long (*RedisAI_NumThreadsPerQueue)(void);
long long (*RedisAI_OnnxTimeout)(void);

int RAI_InitBackendORT(int (*get_api_fn)(const char *, void **));

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                              const char *modeldef, size_t modellen, RAI_Error *err);

void RAI_ModelFreeORT(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunORT(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error);

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionORT(void);
