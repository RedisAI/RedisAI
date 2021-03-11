#ifndef SRC_BACKENDS_ONNXRUNTIME_H_
#define SRC_BACKENDS_ONNXRUNTIME_H_

#include "config.h"
#include "tensor_struct.h"
#include "model_struct.h"
#include "err.h"

unsigned long long RAI_GetMemoryInfoORT(void);

unsigned long long RAI_GetMemoryAccessORT(void);

int RAI_InitBackendORT(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                              const char *modeldef, size_t modellen, RAI_Error *err);

void RAI_ModelFreeORT(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunORT(RAI_ModelRunCtx **mctxs, RAI_Error *error);

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

const char *RAI_GetBackendVersionORT(void);

#endif /* SRC_BACKENDS_ONNXRUNTIME_H_ */
