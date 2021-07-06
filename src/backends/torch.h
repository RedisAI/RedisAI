#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model.h"
#include "redis_ai_objects/script.h"
#include "execution/execution_contexts/execution_ctx.h"

int RAI_InitBackendTorch(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateTorch(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                                const char *modeldef, size_t modellen, RAI_Error *err);

void RAI_ModelFreeTorch(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTorch(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error);

int RAI_ModelSerializeTorch(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

RAI_Script *RAI_ScriptCreateTorch(const char *devicestr, const char *scriptdef,
                                  const char **entry_points, size_t n_entry_points,
                                  RAI_Error *error);

void RAI_ScriptFreeTorch(RAI_Script *script, RAI_Error *error);

int RAI_ScriptRunTorch(RAI_Script *script, const char *function, RAI_ExecutionCtx *ectx,
                       RAI_Error *error);

const char *RAI_GetBackendVersionTorch(void);
