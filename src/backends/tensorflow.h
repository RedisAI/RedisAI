#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/tensor_struct.h"
#include "redis_ai_objects/model_struct.h"

int RAI_InitBackendTF(int (*get_api_fn)(const char *, void *));

int RAI_ModelCreateTF(RAI_Model *model, RAI_Error *error);

void RAI_ModelFreeTF(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTF(RAI_ModelRunCtx **mctxs, RAI_Error *error);

int RAI_ModelSerializeTF(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionTF(void);
