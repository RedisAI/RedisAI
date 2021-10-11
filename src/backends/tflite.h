#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model.h"
#include "execution/execution_contexts/execution_ctx.h"

int RAI_InitBackendTFLite(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateTFLite(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                                 const char *modeldef, size_t modellen, RAI_Error *err);

void RAI_ModelFreeTFLite(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTFLite(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error);

int RAI_ModelSerializeTFLite(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionTFLite(void);
