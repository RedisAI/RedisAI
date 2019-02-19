#ifndef SRC_BACKENDS_TENSORFLOW_H_
#define SRC_BACKENDS_TENSORFLOW_H_

#include "config.h"
#include "tensor_struct.h"
#include "model_struct.h"

#include "tensorflow/c/c_api.h"

RAI_Tensor* RAI_TensorCreateFromTFTensor(TF_Tensor *tensor);

TF_Tensor* RAI_TFTensorFromTensor(RAI_Tensor* t);

RAI_Model *RAI_ModelCreateTF(RAI_Backend backend, RAI_Device device,
                             char **inputs, char **outputs,
                             const char *modeldef, size_t modellen);

void RAI_ModelFreeTF(RAI_Model* model);

int RAI_ModelRunTF(RAI_ModelRunCtx* mctx);

#endif /* SRC_BACKENDS_TENSORFLOW_H_ */
