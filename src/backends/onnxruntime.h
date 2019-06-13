#ifndef SRC_BACKENDS_ONNXRUNTIME_H_
#define SRC_BACKENDS_ONNXRUNTIME_H_

#include "config.h"
#include "tensor_struct.h"
#include "model_struct.h"
#include "err.h"

#include "onnxruntime_c_api.h"

RAI_Model *RAI_ModelCreateORT(RAI_Backend backend, RAI_Device device,
                              const char *modeldef, size_t modellen,
                              RAI_Error *err);

void RAI_ModelFreeORT(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunORT(RAI_ModelRunCtx *mctx, RAI_Error *error);

int RAI_ModelSerializeORT(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

#endif /* SRC_BACKENDS_ONNXRUNTIME_H_ */
