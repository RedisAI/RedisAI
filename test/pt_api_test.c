#include <stdio.h>
#include <torch_c.h>

int main() {
  printf("Hello from libTorch C library version %s\n", "1.0");
  torchBasicTest();

  void* ctx;

  const char script0[] = "\n\
    def foo(a):          \n\
        return _a * 2     \n\
    ";
  ctx = torchCompileScript(script0);
  printf("Should be NULL: %p\n", ctx);

  const char script[] = "\n\
    def foo(a):          \n\
        return a * 2     \n\
    ";
  ctx = torchCompileScript(script);
  printf("Compiled: %p\n", ctx);

  //DLTensor* input;
  //DLTensor* output;
  //torchRunScript(ctx, "foo", 1, &input, 1, &output);

  torchDeallocScript(ctx);
  printf("Deallocated\n");

  return 0;
}
