#ifndef SRC_BACKENDS_TENSORFLOW_H_
#define SRC_BACKENDS_TENSORFLOW_H_

#include "config.h"
#include "tensor_struct.h"
#include "graph_struct.h"

#include "tensorflow/c/c_api.h"

RAI_Tensor* RAI_TensorCreateFromTFTensor(TF_Tensor *tensor);

TF_Tensor* RAI_TFTensorFromTensor(RAI_Tensor* t);

RAI_Graph *RAI_GraphCreateTF(const char *prefix,
                             RAI_Backend backend, RAI_Device device,
                             const char *graphdef, size_t graphlen);

void RAI_GraphFreeTF(RAI_Graph* graph);

int RAI_GraphRunTF(RAI_GraphRunCtx* gctx);

#endif /* SRC_BACKENDS_TENSORFLOW_H_ */
