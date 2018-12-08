#ifndef SRC_TENSOR_STRUCT_H_
#define SRC_TENSOR_STRUCT_H_

#include "config.h"
#include "dlpack/dlpack.h"

typedef struct RDL_Tensor {
  DLTensor tensor;
  long long refCount;
} RDL_Tensor;

#endif /* SRC_TENSOR_STRUCT_H_ */
