/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model.h"
#include "execution/execution_contexts/execution_ctx.h"

int RAI_InitBackendTF(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateTF(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                             size_t ninputs, const char **inputs, size_t noutputs,
                             const char **outputs, const char *modeldef, size_t modellen,
                             RAI_Error *error);

void RAI_ModelFreeTF(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTF(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error);

int RAI_ModelSerializeTF(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionTF(void);
