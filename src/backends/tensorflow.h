#ifndef SRC_BACKENDS_TENSORFLOW_H_
#define SRC_BACKENDS_TENSORFLOW_H_

#include "config.h"
#include "tensor_struct.h"
#include "model_struct.h"
#include "err.h"

#include "tensorflow/c/c_api.h"

RAI_Tensor* RAI_TensorCreateFromTFTensor(TF_Tensor *tensor);

TF_Tensor* RAI_TFTensorFromTensor(RAI_Tensor* t);

RAI_Model *RAI_ModelCreateTF(RAI_Backend backend, RAI_Device device,
                             size_t ninputs, const char **inputs,
                             size_t noutputs, const char **outputs,
                             const char *modeldef, size_t modellen,
                             RAI_Error *error);

void RAI_ModelFreeTF(RAI_Model* model, RAI_Error *error);

int RAI_ModelRunTF(RAI_ModelRunCtx* mctx, RAI_Error *error);

#endif /* SRC_BACKENDS_TENSORFLOW_H_ */
