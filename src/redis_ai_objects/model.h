/**
 * model.h
 *
 * Contains headers for the helper methods for both creating, populating,
 * managing and destructing the RedisModuleType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */
#pragma once

#include "redismodule.h"
#include "redisai.h"
#include "err.h"
#include "tensor.h"
#include "model_struct.h"
#include "config/config.h"
#include "util/dict.h"
#include "execution/run_info.h"

/**
 * Helper method to allocated and initialize a RAI_Model. Depending on the
 * backend it relies on either `model_create_with_nodes` or `model_create`
 * callback functions.
 *
 * @param backend enum identifying the backend. one of RAI_BACKEND_TENSORFLOW,
 * RAI_BACKEND_TFLITE,  RAI_BACKEND_TORCH, or RAI_BACKEND_ONNXRUNTIME
 * @param devicestr device string
 * @param tag optional model tag
 * @param opts `RAI_ModelOpts` like batchsize or parallelism settings
 * @param ninputs optional number of inputs definition
 * @param inputs optional inputs array
 * @param noutputs optional number of outputs
 * @param outputs optional number of outputs array
 * @param modeldef encoded model definition
 * @param modellen length of the encoded model definition
 * @param error error data structure to store error message in the case of
 * failures
 * @return RAI_Model model structure on success, or NULL if failed
 */
RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char *devicestr, RedisModuleString *tag,
                           RAI_ModelOpts opts, size_t ninputs, const char **inputs, size_t noutputs,
                           const char **outputs, const char *modeldef, size_t modellen,
                           RAI_Error *err);

/**
 * Frees the memory of the RAI_Model when the model reference count reaches
 * 0. It is safe to call this function with a NULL input model.
 *
 * @param model input model to be freed
 * @param error error data structure to store error message in the case of
 * failures
 */
void RAI_ModelFree(RAI_Model *model, RAI_Error *err);

/**
 * Every call to this function, will make the RAI_Model 'model' requiring an
 * additional call to RAI_ModelFree() in order to really free the model.
 * Returns a shallow copy of the model.
 *
 * @param input model
 * @return model
 */
RAI_Model *RAI_ModelGetShallowCopy(RAI_Model *model);

/**
 * Serializes a model given the RAI_Model pointer, saving the serialized data
 * into `buffer` and proving the saved buffer size
 *
 * @param model RAI_Model pointer
 * @param buffer pointer to the output buffer
 * @param len pointer to the variable to save the output buffer length
 * @param error error data structure to store error message in the case of
 * failures
 * @return REDISMODULE_OK if the underlying backend `model_serialize` ran
 * successfully, or REDISMODULE_ERR if failed.
 * */
int RAI_ModelSerialize(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err);

/**
 * Helper method to get a Model from keyspace. In the case of failure the key is
 * closed and the error is replied ( no cleaning actions required )
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param model destination model structure
 * @param mode key access mode
 * @param error contains the error in case of problem with retrival
 * @return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable, or REDISMODULE_ERR if there was
 * an error getting the Model
 */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Model **model,
                             int mode, RAI_Error *err);

/**
 * When a module command is called in order to obtain the position of
 * keys, since it was flagged as "getkeys-api" during the registration,
 * the command implementation checks for this special call using the
 * RedisModule_IsKeysPositionRequest() API and uses this function in
 * order to report keys.
 * No real execution is done on this special call.
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return
 */
int RedisAI_ModelRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx, RedisModuleString **argv,
                                                      int argc);

/**
 * See "RedisAI_ModelRun_IsKeysPositionRequest_ReportKeys" above. While this function
 * is used for AI.MODELEXECUTE command, RedisAI_ModelRun_IsKeysPositionRequest_ReportKeys"
 * is used for the deprecated AI.MODELRUN command syntax.
 */
int ModelExecute_ReportKeysPositions(RedisModuleCtx *ctx, RedisModuleString **argv, int argc);

/**
 * @brief  Returns the number of inputs in the model definition.
 */
size_t RAI_ModelGetNumInputs(RAI_Model *model);

/**
 * @brief Returns the input name in a given index.
 */
const char *RAI_ModelGetInputName(RAI_Model *model, size_t index);

/**
 * @brief  Returns the number of outputs in the model definition.
 */
size_t RAI_ModelGetNumOutputs(RAI_Model *model);

/**
 * @brief Returns the output name in a given index.
 */
const char *RAI_ModelGetOutputName(RAI_Model *model, size_t index);

/**
 * @brief Returns the RAI_Model object internal session object.
 */
void *RAI_ModelGetSession(RAI_Model *model);

/**
 * @brief Returns the RAI_Model object internal model object.
 */
void *RAI_ModelGetModel(RAI_Model *model);

/**
 * @brief  Returns the redis module type representing a model.
 * @return redis module type representing a model.
 */
RedisModuleType *RAI_ModelRedisType(void);
