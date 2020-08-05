/**
 * model.h
 *
 * Contains headers for the helper methods for both creating, populating,
 * managing and destructing the RedisModuleType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#ifndef SRC_MODEL_H_
#define SRC_MODEL_H_

#include "config.h"
#include "err.h"
#include "model_struct.h"
#include "redisai.h"
#include "redismodule.h"
#include "run_info.h"
#include "tensor.h"
#include "util/dict.h"

extern RedisModuleType* RedisAI_ModelType;

/**
 * Helper method to register the RedisModuleType type exported by the module.
 *
 * @param ctx Context in which Redis modules operate
 * @return
 */
int RAI_ModelInit(RedisModuleCtx* ctx);

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
RAI_Model* RAI_ModelCreate(RAI_Backend backend, const char* devicestr,
                           const char* tag, RAI_ModelOpts opts, size_t ninputs,
                           const char** inputs, size_t noutputs,
                           const char** outputs, const char* modeldef,
                           size_t modellen, RAI_Error* err);

/**
 * Frees the memory of the RAI_Model when the model reference count reaches
 * 0. It is safe to call this function with a NULL input model.
 *
 * @param model input model to be freed
 * @param error error data structure to store error message in the case of
 * failures
 */
void RAI_ModelFree(RAI_Model* model, RAI_Error* err);

/**
 * Allocates the RAI_ModelRunCtx data structure required for async background
 * work within `RedisAI_RunInfo` structure on RedisAI blocking commands
 *
 * @param model input model
 * @return RAI_ModelRunCtx to be used within
 */
RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model);

/**
 * Frees the RAI_ModelRunCtx data structure used within for async background
 * work
 *
 * @param mctx
 * @param freeTensors free input and output tensors or leave them allocated
 */
void RAI_ModelRunCtxFree(RAI_ModelRunCtx* mctx, int freeTensors);

/**
 * Allocates a RAI_ModelCtxParam data structure, and enforces a shallow copy of
 * the provided input tensor, adding it to the input tensors array of the
 * RAI_ModelRunCtx.
 *
 * @param mctx input RAI_ModelRunCtx to add the input tensor
 * @param inputName input tensor name
 * @param inputTensor input tensor structure
 * @return returns 1 on success ( always returns success )
 */
int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* mctx, const char* inputName,
                            RAI_Tensor* inputTensor);

/**
 * Allocates a RAI_ModelCtxParam data structure, and sets the tensor reference
 * to NULL ( will be set after MODELRUN ), adding it to the outputs tensors
 * array of the RAI_ModelRunCtx.
 *
 * @param mctx RAI_ModelRunCtx to add the output tensor
 * @param outputName output tensor name
 * @return returns 1 on success ( always returns success )
 */
int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* mctx, const char* outputName);

/**
 * Returns the total number of input tensors of the RAI_ModelRunCtx
 *
 * @param mctx RAI_ModelRunCtx
 * @return the total number of input tensors of the RAI_ModelRunCtx
 */
size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx* mctx);

/**
 * Returns the total number of output tensors of the RAI_ModelCtxParam
 *
 * @param mctx RAI_ModelRunCtx
 * @return the total number of output tensors of the RAI_ModelCtxParam
 */
size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* mctx);

/**
 * Get the RAI_Tensor at the input array index position
 *
 * @param mctx RAI_ModelRunCtx
 * @param index input array index position
 * @return RAI_Tensor
 */
RAI_Tensor* RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx* mctx, size_t index);

/**
 * Get the RAI_Tensor at the output array index position
 *
 * @param mctx RAI_ModelRunCtx
 * @param index input array index position
 * @return RAI_Tensor
 */
RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* mctx, size_t index);

/**
 * Given the input array of mctxs, run the associated backend
 * session. If the input array of model context runs is larger than one, then
 * each backend's `model_run` is responsible for concatenating tensors, and run
 * the model in batches with the size of the input array. On success, the
 * tensors corresponding to outputs[0,noutputs-1] are placed in each
 * RAI_ModelRunCtx output tensors array. Relies on each backend's `model_run`
 * definition.
 *
 * @param mctxs array on input model contexts
 * @param n length of input model contexts array
 * @param error error data structure to store error message in the case of
 * failures
 * @return REDISMODULE_OK if the underlying backend `model_run` runned
 * successfully, or REDISMODULE_ERR if failed.
 */
int RAI_ModelRun(RAI_ModelRunCtx** mctxs, long long n, RAI_Error* err);

/**
 * Every call to this function, will make the RAI_Model 'model' requiring an
 * additional call to RAI_ModelFree() in order to really free the model.
 * Returns a shallow copy of the model.
 *
 * @param input model
 * @return model
 */
RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model);

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
int RAI_ModelSerialize(RAI_Model* model, char** buffer, size_t* len,
                       RAI_Error* err);

/**
 * Helper method to get a Model from keyspace. In the case of failure the key is
 * closed and the error is replied ( no cleaning actions required )
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key models's key handle. On success it contains an handle representing
 * a Redis key with the requested access mode
 * @param model destination model structure
 * @param mode key access mode
 * @return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable, or REDISMODULE_ERR if there was
 * an error getting the Model
 */
int RAI_GetModelFromKeyspace(RedisModuleCtx* ctx, RedisModuleString* keyName,
                             RedisModuleKey** key, RAI_Model** model, int mode);

/**
 * Helper method to parse AI.MODELRUN arguments
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param mctx Destination Model context to store the parsed data
 * @param outkeys array to store the parsed output keys
 * @param mto model to run the session from
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RedisAI_Parse_ModelRun_RedisCommand(
    RedisModuleCtx* ctx, RedisModuleString** argv, int argc,
    RAI_ModelRunCtx** mctx, RedisModuleString*** inkeys, RedisModuleString*** outkeys,
    RAI_Model** mto, RAI_Error* error);

/**
 * @brief  Returns the redis module type representing a model.
 * @return redis module type representing a model.
 */
RedisModuleType *RAI_ModelRedisType(void);

#endif /* SRC_MODEL_H_ */
