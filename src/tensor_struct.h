#ifndef SRC_TENSOR_STRUCT_H_
#define SRC_TENSOR_STRUCT_H_

#include "config.h"
#include "dlpack/dlpack.h"

typedef struct RAI_Tensor {
  DLManagedTensor tensor;
  long long refCount;
} RAI_Tensor;

#endif /* SRC_TENSOR_STRUCT_H_ */
