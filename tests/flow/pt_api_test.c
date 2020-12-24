#include <stdlib.h>
#include <stdio.h>
#include <torch_c.h>
#include "dlpack/dlpack.h"

int main() {
  printf("Hello from libTorch C library version %s\n", "1.0");
  torchBasicTest();

  void* ctx;
  char *err = NULL;

  const char script[] = "\n\
    def foo(a):          \n\
        return a * 2     \n\
    ";

  ctx = torchCompileScript(script, kDLCPU, &err);
  if (err) {
    printf("ERR: %s\n", err);
    free(err);
    err = NULL;
    return 1;
  }
  printf("Compiled: %p\n", ctx);

  DLDataType dtype = (DLDataType){ .code = kDLFloat, .bits = 32, .lanes = 1};
  int64_t shape[1] = {1};
  int64_t strides[1] = {1};
  char data[4] = "0000";
  DLManagedTensor* input = torchNewTensor(dtype, 1, shape, strides, data);
  DLManagedTensor* output;
  torchRunScript(ctx, "foo", 1, &input, 1, &output, &err);
  if (err) {
    printf("ERR: %s\n", err);
    free(err);
    return 1;
  }

  torchDeallocContext(ctx);
  printf("Deallocated\n");

  return 0;
}
