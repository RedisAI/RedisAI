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

#include "redis_ai_objects/script_struct.h"

typedef struct TorchFunctionInputCtx {
    DLManagedTensor **tensorInputs;
    size_t tensorCount;
    RedisModuleString **args;
    size_t argsCount;
    RedisModuleString **keys;
    size_t keysCount;
    bool hasEntryPoint; // TODO: remove this when SCRIPTRUN is EOL. Indication that the script was
                        // stored with SCRIPTSET and not SCRIPTSTORE, such that it has no entry
                        // point, so execution is best effort.
} TorchFunctionInputCtx;

/**
 * @brief Register Redis ops in Torch
 *
 */
void torchRegisterRedisOps(void);

/**
 * @brief Compiles a script string into torch compliation unit stored in a module context.
 *
 * @param script Script string.
 * @param device Device for the script to execute on.
 * @param device_id Device id for the script to execute on.
 * @param error Error string to be populated in case of an exception.
 * @return void* ModuleContext pointer.
 */
void *torchCompileScript(const char *script, DLDeviceType device, int64_t device_id, char **error);

/**
 * @brief Loads a model from model definition string and stores it in a module context.
 *
 * @param model Model definition string.
 * @param modellen Length of the string.
 * @param device Device for the model to execute on.
 * @param device_id Device id for the model to execute on.
 * @param error Error string to be populated in case of an exception.
 * @return void* ModuleContext pointer.
 */
void *torchLoadModel(const char *model, size_t modellen, DLDeviceType device, int64_t device_id,
                     char **error);

/**
 * @brief Executes a function in a script.
 * @param scriptCtx Executes a function in a script.
 * @param fnName Function name.
 * @param inputsCtx unction execution context containing the information about given inputs.
 * @param outputs Array of output tensor (placeholders).
 * @param nOutputs Number of output tensors.
 * @param error Error string to be populated in case of an exception.
 */
void torchRunScript(void *scriptCtx, const char *fnName, TorchFunctionInputCtx *inputsCtx,
                    DLManagedTensor **outputs, long nOutputs, char **error);

/**
 * @brief Executes a model.
 *
 * @param modelCtx Model context.
 * @param nInputs Number of tensor inputs.
 * @param inputs Array of input tensors.
 * @param nOutputs Number of output tensors.
 * @param outputs Array of output tensor (placeholders).
 * @param error Error string to be populated in case of an exception.
 */
void torchRunModel(void *modelCtx, long nInputs, DLManagedTensor **inputs, long nOutputs,
                   DLManagedTensor **outputs, char **error);

/**
 * @brief
 *
 * @param modelCtx Serilized a model into a string defintion.
 * @param buffer Byte array to hold the definition.
 * @param len Will store the length of the string.
 * @param error Error string to be populated in case of an exception.
 */
void torchSerializeModel(void *modelCtx, char **buffer, size_t *len, char **error);

/**
 * @brief Deallicate the create torch script/model object.
 *
 * @param ctx Object to free.
 */
void torchDeallocContext(void *ctx);

/**
 * @brief Sets the number of inter-op threads for Torch backend.
 *
 * @param num_threads Number of inter-op threads.
 * @param error Error string to be populated in case of an exception.
 */
void torchSetInterOpThreads(int num_threads, char **error);

/**
 * @brief Sets the number of intra-op threads for Torch backend.
 *
 * @param num_threads Number of intra-op threads.
 * @param error Error string to be populated in case of an exception.
 */
void torchSetIntraOpThreads(int num_threadsm, char **error);

/**
 * @brief Returns the number of inputs of a model
 *
 * @param modelCtx Model context.
 * @param error Error string to be populated in case of an exception.
 * @return size_t Number of inputs.
 */
size_t torchModelNumInputs(void *modelCtx, char **error);

/**
 * @brief Returns the name of the model input at index.
 *
 * @param modelCtx Model context.
 * @param index Input index.
 * @param error Error string to be populated in case of an exception.
 * @return const char* Input name.
 */
const char *torchModelInputNameAtIndex(void *modelCtx, size_t index, char **error);

/**
 * @brief Returns the number of outputs of a model
 *
 * @param modelCtx Model context.
 * @param error Error string to be populated in case of an exception.
 * @return size_t Number of outputs.
 */
size_t torchModelNumOutputs(void *modelCtx, char **error);

/**
 * @brief Return the number of functions in the script.
 *
 * @param scriptCtx Script context.
 * @return size_t number of functions.
 */
size_t torchScript_FunctionCount(void *scriptCtx);

/**
 * @brief Return the name of the function numbered fn_index in the script.
 *
 * @param scriptCtx Script context.
 * @param fn_index Function number.
 * @return const char* Function name.
 */
const char *torchScript_FunctionName(void *scriptCtx, size_t fn_index);

/**
 * @brief Return the number of arguments of a given fuction in the script.
 *
 * @param scriptCtx Script context.
 * @param functionName Function name.
 * @return size_t Number of arguments.
 */
size_t torchScript_FunctionArgumentCountByFunctionName(void *scriptCtx, const char *functionName);

/**
 * @brief Returns the type of the argument at arg_index of a given function in the
 * script.
 *
 * @param scriptCtx Script context.
 * @param functionName Function name.
 * @param arg_index Argument number.
 * @return TorchScriptFunctionArgumentType The type of the argument in RedisAI enum format.
 */
TorchScriptFunctionArgumentType
torchScript_FunctionArgumentTypeByFunctionName(void *scriptCtx, const char *functionName,
                                               size_t arg_index);

/**
 * @brief Returns if function with a given name exists in the script
 *
 * @param scriptCtx Script context.
 * @param functionName Function name.
 * @return true If the function exists.
 * @return false If the function does not exists.
 */
bool torchScript_FunctionExists(void *scriptCtx, const char *functionName);

/**
 * @brief Creates a new dltensor representation from torch tensor, by taking
 * ownership on the tensor and keeping it in the manager_context field. The tensor
 * data will be freed by calling the deleter function on the manager context field.
 * @param src - A pointer to torch tensor.
 * @returns The newly created DLManaged tensor.
 */
DLManagedTensor *torchTensorPtrToManagedDLPack(const void *src);

/**
 * @brief Creates a new torch tensor from a RedisAI tensor, by using its data
 * and store it in torch_tensor pointer. Note that the ownership of the tensor
 * is transferred to the torch tensor, and it will be released by calling the
 * created deleter function, which is RAI_TensorFree
 * @param src - the input RAI tensor
 * @param torch_tensor - place holder for the newly created torch tensor.
 */
void torchTensorFromRAITensor(RAI_Tensor *src, void *torch_tensor);

#ifdef __cplusplus
}
#endif
