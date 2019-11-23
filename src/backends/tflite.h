#ifndef SRC_BACKENDS_TFLITE_H_
#define SRC_BACKENDS_TFLITE_H_

#include "config.h"
#include "tensor_struct.h"
#include "model_struct.h"
#include "err.h"

int RAI_InitBackendTFLite(int (*get_api_fn)(const char *, void *));

RAI_Model *RAI_ModelCreateTFLite(RAI_Backend backend, const char* devicestr,
                                 const char *modeldef, size_t modellen,
                                 RAI_Error *err);

void RAI_ModelFreeTFLite(RAI_Model *model, RAI_Error *error);

int RAI_ModelRunTFLite(RAI_ModelRunCtx *mctx, RAI_Error *error);

int RAI_ModelSerializeTFLite(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error);

#endif /* SRC_BACKENDS_TFLITE_H_ */
