#ifndef SRC_BACKENDS_TENSORFLOW_H_
#define SRC_BACKENDS_TENSORFLOW_H_

#include "config.h"
#include "tensor_struct.h"
#include "graph_struct.h"

#include "tensorflow/c/c_api.h"

RDL_Tensor* RDL_TensorCreateFromTFTensor(TF_Tensor *tensor);

TF_Tensor* RDL_TFTensorFromTensor(RDL_Tensor* t);

RDL_Graph *RDL_GraphCreateTF(const char *prefix, RDL_Backend backend,
                             const char *graphdef, size_t graphlen);

void RDL_GraphFreeTF(RDL_Graph* graph);

int RDL_GraphRunTF(RDL_GraphRunCtx* gctx);

#endif /* SRC_BACKENDS_TENSORFLOW_H_ */