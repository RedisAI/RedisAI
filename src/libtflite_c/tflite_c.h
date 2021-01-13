#ifndef __TFLITE_C_H__
#define __TFLITE_C_H__

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

// void tfliteBasicTest();

void *tfliteLoadModel(const char *model, size_t modellen, DLDeviceType device, int64_t device_id,
                      char **error);

void tfliteRunModel(void *ctx, long nInputs, DLManagedTensor **inputs, long nOutputs,
                    DLManagedTensor **outputs, char **error);

void tfliteSerializeModel(void *ctx, char **buffer, size_t *len, char **error);

void tfliteDeallocContext(void *ctx);

#ifdef __cplusplus
}
#endif

#endif // __TFLITE_C_H__
