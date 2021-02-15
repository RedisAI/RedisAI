#pragma once

#include "config.h"
#include "tensor_struct.h"
#include "model_struct.h"
#include "err.h"

int RAI_InitBackendTFLite(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateTFLite(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                                 const char *modeldef, size_t modellen, RAI_Error *err);

void RAI_ModelFreeTFLite(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTFLite(RAI_ModelRunCtx **mctxs, RAI_Error *error);

int RAI_ModelSerializeTFLite(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionTFLite(void);
