#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model_struct.h"
#include "redis_ai_objects/tensor_struct.h"

int RAI_InitBackendTFLite(int (*get_api_fn)(const char *, void *));

int RAI_ModelCreateTFLite(RAI_Model *model, RAI_Error *err);

void RAI_ModelFreeTFLite(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTFLite(RAI_ModelRunCtx **mctxs, RAI_Error *error);

int RAI_ModelSerializeTFLite(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionTFLite(void);
