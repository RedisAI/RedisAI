#pragma once

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "redis_ai_objects/script_struct.h"

typedef struct TorchFunctionInputCtx {
    DLManagedTensor **tensorInputs;
    size_t tensorCount;
    int32_t *intInputs;
    size_t intCount;
    float *floatInputs;
    size_t floatCount;
    RedisModuleString **stringsInputs;
    size_t stringCount;
    size_t *listSizes;
    size_t listCount;
} TorchFunctionInputCtx;

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
 * @brief  Validate SCRIPTEXECUTE or LLAPI script execute inputs according to the funciton schema.
 *
 * @param schema Fuction argument types (schema).
 * @param nArguments Number of arguments in the function.
 * @param inputsCtx Function execution context containing the information about given inputs.
 * @param error Error string to be populated in case of an exception.
 * @return true If the user provided inputs from types and order that matches the schema.
 * @return false Otherwise.
 */
bool torchMatchScriptSchema(TorchScriptFunctionArgumentType *schema, size_t nArguments,
                            TorchFunctionInputCtx *inputsCtx, char **error);

/**
 * @brief Executes a function in a script.
 * @note Should be called after torchMatchScriptSchema verication.
 * @param scriptCtx Executes a function in a script.
 * @param fnName Function name.
 * @param schema Fuction argument types (schema).
 * @param nArguments Number of arguments in the function.
 * @param inputsCtx unction execution context containing the information about given inputs.
 * @param outputs Array of output tensor (placeholders).
 * @param nOutputs Number of output tensors.
 * @param error Error string to be populated in case of an exception.
 */
void torchRunScript(void *scriptCtx, const char *fnName, TorchScriptFunctionArgumentType *schema,
                    size_t nArguments, TorchFunctionInputCtx *inputsCtx, DLManagedTensor **outputs,
                    long nOutputs, char **error);

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
 * @brief Return the number of arguments in the fuction numbered fn_index in the script.
 *
 * @param scriptCtx Script context.
 * @param fn_index Function number.
 * @return size_t Number of arguments.
 */
size_t torchScript_FunctionArgumentCount(void *scriptCtx, size_t fn_index);

/**
 * @brief Rerturns the type of the argument at arg_index of function numbered fn_index in the
 * script.
 *
 * @param scriptCtx Script context.
 * @param fn_index Function number.
 * @param arg_index Argument number.
 * @return TorchScriptFunctionArgumentType The type of the argument in RedisAI enum format.
 */
TorchScriptFunctionArgumentType torchScript_FunctionArgumentype(void *scriptCtx, size_t fn_index,
                                                                size_t arg_index);

DLManagedTensor *torchTensorPtrToManagedDLPack(const void *src);

/**
 * @brief Creates a new torch tensor from a tensor represented in dlpack by copying
 * the data, and storing it in torch_tensor pointer. Note that the data will be
 * freed by calling the given deleter function, which is RedisModule_Free
 * @param src - the input dlpack tensor
 * @param torch_tensor - place holder for the newly created torch tensor.
 * @param data_len - the number of bytes in the tensor blob
 * @return
 */
void torchTensorCopyDataFromDLPack(const DLTensor *src, void *torch_tensor, size_t data_len);

#ifdef __cplusplus
}
#endif
