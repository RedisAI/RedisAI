/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

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

size_t tfliteModelNumInputs(void *ctx, char **error);

const char *tfliteModelInputNameAtIndex(void *modelCtx, size_t index, char **error);

size_t tfliteModelNumOutputs(void *ctx, char **error);

const char *tfliteModelOutputNameAtIndex(void *modelCtx, size_t index, char **error);

#ifdef __cplusplus
}
#endif
