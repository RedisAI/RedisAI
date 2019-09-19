#ifndef SRC_BACKENDS_TORCH_H_
#define SRC_BACKENDS_TORCH_H_

#include "config.h"
#include "tensor_struct.h"
#include "script_struct.h"
#include "model_struct.h"
#include "err.h"

int RAI_InitBackendTorch(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateTorch(RAI_Backend backend, RAI_Device device, int64_t deviceid, const char* devicestr,
                                const char *modeldef, size_t modellen,
                                RAI_Error *err);

void RAI_ModelFreeTorch(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTorch(RAI_ModelRunCtx *mctx, RAI_Error *error);

int RAI_ModelSerializeTorch(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

RAI_Script *RAI_ScriptCreateTorch(RAI_Device device, int64_t deviceid, const char* devicestr, const char *scriptdef,
                                  RAI_Error *error);

void RAI_ScriptFreeTorch(RAI_Script *script, RAI_Error *error);

int RAI_ScriptRunTorch(RAI_ScriptRunCtx *sctx, RAI_Error *error);

#endif /* SRC_BACKENDS_TORCH_H_ */
