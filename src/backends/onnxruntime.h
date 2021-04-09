#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/tensor_struct.h"
#include "redis_ai_objects/model_struct.h"

unsigned long long RAI_GetMemoryInfoORT(void);

unsigned long long RAI_GetMemoryAccessORT(void);

int RAI_InitBackendORT(int (*get_api_fn)(const char *, void *));

int RAI_ModelCreateORT(RAI_Model *model, RAI_Error *err);

void RAI_ModelFreeORT(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunORT(RAI_ModelRunCtx **mctxs, RAI_Error *error);

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionORT(void);
