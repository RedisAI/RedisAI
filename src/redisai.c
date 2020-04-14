#include "redismodule.h"
#include "tensor.h"

#include "model.h"
#include "dag.h"
#include "model_script_run_session.h"
#include "background_workers.h"
#include "script.h"
#include "backends.h"
#include "stats.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>

#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"
#include "rmutil/args.h"
#include "run_info.h"

#define REDISAI_H_INCLUDE
#include "redisai.h"
#undef REDISAI_H_INCLUDE


/* ----------------------- RedisAI Module Commands ------------------------- */

/**
 * AI.TENSORSET key type dim1..dimN [BLOB data | VALUES val1..valN]
 */
int RedisAI_TensorSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key;
  const int status = RAI_OpenKey_Tensor(ctx, argv[1], &key, REDISMODULE_READ|REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RAI_Tensor *t=NULL;
  RAI_Error err;
  const int parse_result = RAI_parseTensorSetArgs(ctx,argv,argc,&t,1,&err);

  // if the number of parsed args is negative something went wrong
  if(parse_result<0){
    RedisModule_CloseKey(key);
    return REDISMODULE_ERR;
  }

  if( RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, t) != REDISMODULE_OK ){
    RAI_TensorFree(t);
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
  }
  RedisModule_CloseKey(key);
  RedisModule_ReplyWithSimpleString(ctx, "OK");
  RedisModule_ReplicateVerbatim(ctx);
  return REDISMODULE_OK;
}

/**
* AI.TENSORGET tensor_key [BLOB | VALUES | META]
*/
int RedisAI_TensorGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 3) return RedisModule_WrongArity(ctx);

  RAI_Tensor *t;
  RedisModuleKey *key;
  const int status = RAI_GetTensorFromKeyspace(ctx, argv[1], &key, &t, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
  return REDISMODULE_ERR;
  }

  const int parse_result = RAI_parseTensorGetArgs(ctx, argv, argc, t);
  RedisModule_CloseKey(key);
  // if the number of parsed args is negative something went wrong
  if(parse_result<0){
    return REDISMODULE_ERR;
  }
  return REDISMODULE_OK;
}

/**
* AI.MODELSET model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]] [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob
*/
int RedisAI_ModelSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 4) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* bckstr;
  int backend;
  AC_GetString(&ac, &bckstr, NULL, 0); 
  if (strcasecmp(bckstr, "TF") == 0) {
    backend = RAI_BACKEND_TENSORFLOW;
  }
  else if (strcasecmp(bckstr, "TFLITE") == 0) {
    backend = RAI_BACKEND_TFLITE;
  }
  else if (strcasecmp(bckstr, "TORCH") == 0) {
    backend = RAI_BACKEND_TORCH;
  }
  else if (strcasecmp(bckstr, "ONNX") == 0) {
    backend = RAI_BACKEND_ONNXRUNTIME;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  if (strlen(devicestr) > 10) {
    return RedisModule_ReplyWithError(ctx, "ERR Invalid DEVICE");
  }

  const char* tag = "";
  if (AC_AdvanceIfMatch(&ac, "TAG")) {
    AC_GetString(&ac, &tag, NULL, 0);
  }

  unsigned long long batchsize = 0;
  if (AC_AdvanceIfMatch(&ac, "BATCHSIZE")) {
    if (backend == RAI_BACKEND_TFLITE) {
      return RedisModule_ReplyWithError(ctx, "ERR Auto-batching not supported by the TFLITE backend");
    }
    if (AC_GetUnsignedLongLong(&ac, &batchsize, 0) != AC_OK) {
      return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for BATCHSIZE");
    }
  }

  unsigned long long minbatchsize = 0;
  if (AC_AdvanceIfMatch(&ac, "MINBATCHSIZE")) {
    if (batchsize == 0) {
      return RedisModule_ReplyWithError(ctx, "ERR MINBATCHSIZE specified without BATCHSIZE");
    }
    if (AC_GetUnsignedLongLong(&ac, &minbatchsize, 0) != AC_OK) {
      return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for MINBATCHSIZE");
    }
  }


  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
  }

  ArgsCursor optionsac;
  AC_GetSliceToOffset(&ac, &optionsac, argc-2);

  if (optionsac.argc == 0 && backend == RAI_BACKEND_TENSORFLOW) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, INPUTS and OUTPUTS not specified");
  }

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};
  if (optionsac.argc > 0) {
    if (!AC_AdvanceIfMatch(&optionsac, "INPUTS")) {
      return RedisModule_ReplyWithError(ctx, "ERR INPUTS not specified");
    }

    const char* matches[] = {"OUTPUTS"};
    AC_GetSliceUntilMatches(&optionsac, &inac, 1, matches);

    if (!AC_IsAtEnd(&optionsac)) {
      if (!AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
        return RedisModule_ReplyWithError(ctx, "ERR OUTPUTS not specified");
      }

      AC_GetSliceToEnd(&optionsac, &outac);
    }
  }

  size_t ninputs = inac.argc;
  const char *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetString(&inac, inputs+i, NULL, 0); 
  }

  size_t noutputs = outac.argc;
  const char *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetString(&outac, outputs+i, NULL, 0); 
  }

  RAI_ModelOpts opts = {
    .batchsize = batchsize,
    .minbatchsize = minbatchsize
  };

  RAI_Model *model = NULL;

  size_t modellen;
  const char *modeldef;
  AC_GetString(&ac, &modeldef, &modellen, 0); 

  RAI_Error err = {0};

  model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "backend %s not loaded, will try loading default backend\n", bckstr);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "could not load %s default backend\n", bckstr);
      int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);
  }

  if (err.code != RAI_OK) {
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  // TODO: if backend loaded, make sure there's a queue
  RunQueueInfo *run_queue_info = NULL;
  if (ensureRunQueue(devicestr,&run_queue_info) != REDISMODULE_OK){
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_ModelType, model);

  model->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_MODEL, backend, devicestr, tag);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

/**
* AI.MODELGET model_key [META | BLOB]
*/
int RedisAI_ModelGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2 && argc != 3) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *key;
  const int status = RAI_GetModelFromKeyspace( ctx, argv[1], &key, &mto, REDISMODULE_READ | REDISMODULE_WRITE);
  if (status == REDISMODULE_ERR) {
    return REDISMODULE_ERR;
  }
  int blob = 0;
  if(argc==3){
    const char *optstr = RedisModule_StringPtrLen(argv[2], NULL);
    if (!strcasecmp(optstr, "META")) {
      blob = 0;
    }
    else if (!strcasecmp(optstr, "BLOB")) {
      blob = 1;
    }
  }

  RAI_Error err = {0};

  char *buffer = NULL;
  size_t len = 0;

  if (blob) {
    RAI_ModelSerialize(mto, &buffer, &len, &err);

    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail);
      RAI_ClearError(&err);
      if (*buffer) {
        RedisModule_Free(buffer);
      }
      return ret;
    }
  }

  int outentries = blob ? 8 : 6;

  RedisModule_ReplyWithArray(ctx, outentries);

  RedisModule_ReplyWithSimpleString(ctx, "BACKEND");
  const char* backendstr = RAI_BackendName(mto->backend);
  RedisModule_ReplyWithSimpleString(ctx, backendstr);

  RedisModule_ReplyWithSimpleString(ctx, "DEVICE");
  RedisModule_ReplyWithSimpleString(ctx, mto->devicestr);

  RedisModule_ReplyWithSimpleString(ctx, "TAG");
  RedisModule_ReplyWithSimpleString(ctx, mto->tag ? mto->tag : "");

  if (blob) {
    RedisModule_ReplyWithSimpleString(ctx, "BLOB");
    RedisModule_ReplyWithStringBuffer(ctx, buffer, len);
    RedisModule_Free(buffer);
  }
  RedisModule_CloseKey(key);
  return REDISMODULE_OK;
}

/**
* AI.MODELDEL model_key
*/
int RedisAI_ModelDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *key;
  const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &key, &mto, REDISMODULE_READ|REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);
  RedisModule_ReplicateVerbatim(ctx);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/** 
* AI._MODELLIST
*/
int RedisAI_ModelList_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 1) return RedisModule_WrongArity(ctx);

  RedisModule_Log(ctx, "warning", "MODELLIST is experimental and might be removed in future versions");

  long long nkeys;
  RedisModuleString** keys;
  const char** tags;
  RAI_ListStatsEntries(RAI_MODEL, &nkeys, &keys, &tags);

  RedisModule_ReplyWithArray(ctx, nkeys);

  for (long long i=0; i<nkeys; i++) {
    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithString(ctx, keys[i]);
    RedisModule_ReplyWithSimpleString(ctx, tags[i]);
  }

  RedisModule_Free(keys);
  RedisModule_Free(tags);

  return REDISMODULE_OK;
}

/**
 * AI.MODELRUN model_key INPUTS input_key1 ... OUTPUTS output_key1 ...
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 *
 * 1. clone inputs as needed in the main thread (only the alternative is to
 * lock)
 * 2. spawn the new thread for running the model
 * 3. have reply callback put the data back into the key
 * 
 * This way we avoid any race condition. The only gotcha is making sure no one
 * overwrites the model until it's done computing.
 * This means that setModel will decode on a candidate pointer, and will then
 * be picked up on the next round. We also need to signal when it's time to
 * dispose of the old model. The key is having a single thread looping
 * forexecution
 */
int RedisAI_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                  int argc) {
  if (argc < 3) return RedisModule_WrongArity(ctx);

  RedisAI_RunInfo *rinfo = NULL;
  if (runInfoInit(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
  }

  RAI_Model *mto;
  RedisModuleKey *modelKey;
  const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &modelKey, &mto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }
  
  RedisModule_RetainString(NULL, argv[1]);
  rinfo->runkey = argv[1];
  rinfo->mctx = RAI_ModelRunCtxCreate(mto);

  const int parse_result = RedisAI_Parse_ModelRun_RedisCommand(ctx, argv,
                                   argc, &rinfo->mctx, &rinfo->outkeys, &mto, 0, NULL, 0, NULL, NULL);
  RedisModule_CloseKey(modelKey);
  // if the number of parsed args is negative something went wrong
  if(parse_result<0){
    return REDISMODULE_ERR;
  }

  RunQueueInfo *run_queue_info = NULL;
    // If the queue does not exist, initialize it
  if (ensureRunQueue(mto->devicestr,&run_queue_info) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RAI_ModelRunScriptRunReply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}

/** 
* AI.SCRIPTRUN script_key fn_name INPUTS input_key1 ... OUTPUTS output_key1 ...
*/
int RedisAI_ScriptRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  if (RedisModule_IsKeysPositionRequest(ctx)) {
    RedisModule_KeyAtPos(ctx, 1);
    for (int i=3; i<argc; i++) {
      const char* arg = RedisModule_StringPtrLen(argv[i], NULL);
      if (strcasecmp(arg, "INPUTS") == 0 || strcasecmp(arg, "OUTPUTS") == 0) {
        continue;
      }
      RedisModule_KeyAtPos(ctx, i);
    }
    return REDISMODULE_OK;
  }

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  const char* fnname;
  AC_GetString(&ac, &fnname, NULL, 0); 

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};

  if (!AC_AdvanceIfMatch(&ac, "INPUTS")) {
    return RedisModule_ReplyWithError(ctx, "INPUTS not specified");
  }

  const char* matches[] = {"OUTPUTS"};
  AC_GetSliceUntilMatches(&ac, &inac, 1, matches);

  if (!AC_AdvanceIfMatch(&ac, "OUTPUTS")) {
    return RedisModule_ReplyWithError(ctx, "OUTPUTS not specified");
  }

  AC_GetSliceToEnd(&ac, &outac);

  size_t ninputs = inac.argc;
  RedisModuleString *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetRString(&inac, inputs+i, 0); 
  }

  size_t noutputs = outac.argc;
  RedisModuleString *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetRString(&outac, outputs+i, 0); 
  }

  RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(sto, fnname);

  RedisModuleString **outkeys;

  for (size_t i=0; i<ninputs; i++) {
    RAI_Tensor *t;
    RedisModuleKey *argkey;
    const int status = RAI_GetTensorFromKeyspace(ctx, inputs[i], &argkey, &t, REDISMODULE_READ);
    if(status==REDISMODULE_ERR){
         RedisModule_CloseKey(key);
        return REDISMODULE_ERR;
    }
    RedisModule_CloseKey(argkey);
    if (!RAI_ScriptRunCtxAddInput(sctx, t)) {
      RAI_ScriptRunCtxFree(sctx);
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "Input key not found");
    }
  }

  outkeys = RedisModule_Calloc(noutputs, sizeof(RedisModuleString*));
  for (size_t i=0; i<noutputs; i++) {
    if (!RAI_ScriptRunCtxAddOutput(sctx)) {
      RAI_ScriptRunCtxFree(sctx);
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "Output key not found");
    }
    RedisModule_RetainString(ctx, outputs[i]);
    outkeys[i] = outputs[i];
  }

  RedisAI_RunInfo *rinfo = RedisModule_Calloc(1, sizeof(struct RedisAI_RunInfo));
  rinfo->mctx = NULL;
  rinfo->sctx = sctx;
  RedisModule_RetainString(ctx, keystr);
  rinfo->runkey = keystr;
  rinfo->outkeys = outkeys;
  rinfo->err = NULL;
  RunQueueInfo *run_queue_info = NULL;
    // If the queue does not exist, initialize it
  if (ensureRunQueue(sto->devicestr,&run_queue_info) == REDISMODULE_ERR) {
    RAI_ScriptRunCtxFree(sctx);
    return RedisModule_ReplyWithError(ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RAI_ModelRunScriptRunReply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  RedisModule_ReplicateVerbatim(ctx);
  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}

/**
 * AI.SCRIPTGET script_key
 */
int RedisAI_ScriptGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_ReplyWithArray(ctx, 6);
  RedisModule_ReplyWithSimpleString(ctx, "DEVICE");
  RedisModule_ReplyWithSimpleString(ctx, sto->devicestr);
  RedisModule_ReplyWithSimpleString(ctx, "TAG");
  RedisModule_ReplyWithSimpleString(ctx, sto->tag);
  RedisModule_ReplyWithSimpleString(ctx, "SOURCE");
  RedisModule_ReplyWithSimpleString(ctx, sto->scriptdef);
  RedisModule_CloseKey(key);
  return REDISMODULE_OK;
}

/**
 * AI.SCRIPTDEL script_key
 */
int RedisAI_ScriptDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);

  RedisModule_ReplicateVerbatim(ctx);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/** 
* AI.SCRIPTSET script_key device [TAG tag] script_source
*/
int RedisAI_ScriptSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 4 && argc != 6) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  const char* tag = "";
  if (AC_AdvanceIfMatch(&ac, "TAG")) {
    AC_GetString(&ac, &tag, NULL, 0);
  }

  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "Insufficient arguments, missing script definition");
  }

  RAI_Script *script = NULL;

  size_t scriptlen;
  const char *scriptdef;
  AC_GetString(&ac, &scriptdef, &scriptlen, 0); 

  RAI_Error err = {0};
  script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "Backend TORCH not loaded, will try loading default backend");
    int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load TORCH default backend");
      int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);
  }

  if (err.code != RAI_OK){
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  RunQueueInfo *run_queue_info = NULL;
  // If the queue does not exist, initialize it
  if (ensureRunQueue(devicestr,&run_queue_info) == REDISMODULE_ERR) {
    RAI_ScriptFree(script, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_ScriptType, script);

  script->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

/** 
* AI._SCRIPTLIST
*/
int RedisAI_ScriptList_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 1) return RedisModule_WrongArity(ctx);

  RedisModule_Log(ctx, "warning", "SCRIPTLIST is experimental and might be removed in future versions");

  long long nkeys;
  RedisModuleString** keys;
  const char** tags;
  RAI_ListStatsEntries(RAI_SCRIPT, &nkeys, &keys, &tags);

  RedisModule_ReplyWithArray(ctx, nkeys);

  for (long long i=0; i<nkeys; i++) {
    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithString(ctx, keys[i]);
    RedisModule_ReplyWithSimpleString(ctx, tags[i]);
  }

  RedisModule_Free(keys);
  RedisModule_Free(tags);

  return REDISMODULE_OK;
}

/** 
* AI.INFO <model_or_script_key> [RESETSTAT]
*/
int RedisAI_Info_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 2 && argc != 3) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  const char* runkey;
  AC_GetString(&ac, &runkey, NULL, 0); 

  AI_dictEntry *stats_entry = AI_dictFind(run_stats, runkey);

  if (!stats_entry) {
    return RedisModule_ReplyWithError(ctx, "ERR cannot find run info for key");
  }

  struct RedisAI_RunStats *rstats = AI_dictGetVal(stats_entry);

  if (!AC_IsAtEnd(&ac)) {
    const char* opt;
    AC_GetString(&ac, &opt, NULL, 0); 

    if (strcasecmp(opt, "RESETSTAT") == 0) {
      rstats->duration_us = 0;
      rstats->samples = 0;
      rstats->calls = 0;
      rstats->nerrors = 0;
      RedisModule_ReplyWithSimpleString(ctx, "OK");
      return REDISMODULE_OK;
    }
  }

  RedisModule_ReplyWithArray(ctx, 18);

  RedisModule_ReplyWithSimpleString(ctx, "KEY");
  RedisModule_ReplyWithString(ctx, rstats->key);
  RedisModule_ReplyWithSimpleString(ctx, "TYPE");
  if (rstats->type == 0) {
    RedisModule_ReplyWithSimpleString(ctx, "MODEL");
  }
  else {
    RedisModule_ReplyWithSimpleString(ctx, "SCRIPT");
  }
  RedisModule_ReplyWithSimpleString(ctx, "BACKEND");
  RedisModule_ReplyWithSimpleString(ctx, RAI_BackendName(rstats->backend));
  RedisModule_ReplyWithSimpleString(ctx, "DEVICE");
  RedisModule_ReplyWithSimpleString(ctx, rstats->devicestr);
  RedisModule_ReplyWithSimpleString(ctx, "TAG");
  RedisModule_ReplyWithSimpleString(ctx, rstats->tag);
  RedisModule_ReplyWithSimpleString(ctx, "DURATION");
  RedisModule_ReplyWithLongLong(ctx, rstats->duration_us);
  RedisModule_ReplyWithSimpleString(ctx, "SAMPLES");
  if (rstats->type == 0) {
    RedisModule_ReplyWithLongLong(ctx, rstats->samples);
  }
  else {
    RedisModule_ReplyWithLongLong(ctx, -1);
  }
  RedisModule_ReplyWithSimpleString(ctx, "CALLS");
  RedisModule_ReplyWithLongLong(ctx, rstats->calls);
  RedisModule_ReplyWithSimpleString(ctx, "ERRORS");
  RedisModule_ReplyWithLongLong(ctx, rstats->nerrors);

  return REDISMODULE_OK;
}

int RedisAI_Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  const char* backend;
  AC_GetString(&ac, &backend, NULL, 0); 

  const char* path;
  AC_GetString(&ac, &path, NULL, 0); 

  int ret;
  if (strcasecmp(backend, "TF") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_TENSORFLOW, path);
  }
  else if (strcasecmp(backend, "TFLITE") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_TFLITE, path);
  }
  else if (strcasecmp(backend, "TORCH") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_TORCH, path);
  }
  else if (strcasecmp(backend, "ONNX") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_ONNXRUNTIME, path);
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  if (ret == REDISMODULE_OK) {
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
  }

  return RedisModule_ReplyWithError(ctx, "ERR error loading backend");
}


int RedisAI_Config_BackendsPath(RedisModuleCtx *ctx, const char *path) {
  RedisModule_AutoMemory(ctx);

  if (RAI_BackendsPath != NULL) {
    RedisModule_Free(RAI_BackendsPath);
  }
  RAI_BackendsPath = RedisModule_Strdup(path);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int RedisAI_Config_QueueThreads(RedisModuleString *queueThreadsString) {
  int result = RedisModule_StringToLongLong(queueThreadsString, &perqueueThreadPoolSize);
  // make sure the number of threads is a positive integer
  // if not set the value to the default 
  if (result == REDISMODULE_OK && perqueueThreadPoolSize < 1 ){
    perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;
    result = REDISMODULE_ERR;
  }
  return result;
}

/** 
* AI.CONFIG [BACKENDSPATH <default_location_of_backend_libraries> | LOADBACKEND <backend_identifier> <location_of_backend_library>]
*/
int RedisAI_Config_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  const char* subcommand;
  AC_GetString(&ac, &subcommand, NULL, 0); 

  if (strcasecmp(subcommand, "LOADBACKEND") == 0) {
    return RedisAI_Config_LoadBackend(ctx, argv + 1, argc - 1);
  }

  if (strcasecmp(subcommand, "BACKENDSPATH") == 0) {
    if (argc > 2) {
      return RedisAI_Config_BackendsPath(ctx, RedisModule_StringPtrLen(argv[2], NULL));
    } else {
      return RedisModule_ReplyWithError(ctx, "ERR BACKENDSPATH: missing path argument");
    }
  }

  return RedisModule_ReplyWithError(ctx, "ERR unsupported subcommand");
}

/**
 * AI.DAGRUN [LOAD <nkeys> key1 key2... ] [PERSIST <nkeys> key1 key2... ] |>
 * [COMMAND1] |> [COMMAND2] |> [COMMANDN]
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisAI_RunInfo *rinfo = NULL;
  if (runInfoInit(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
  }
  rinfo->use_local_context = 1;
  RAI_DagOp* currentDagOp = NULL;
  dagInit(&currentDagOp);
  array_append(rinfo->dagOps,currentDagOp);

  int persistFlag=0;
  int loadFlag=0;
  int chainingOpCount=0;
  for (size_t argpos = 1; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "LOAD")) {
      loadFlag=1;
      const int parse_result = RAI_parseDAGLoadArgs(
          ctx, &argv[argpos], argc - argpos, &(rinfo->dagTensorsContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        // TODO: clean temp structures
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "PERSIST")) {
      persistFlag=1;
      const int parse_result =
          RAI_parseDAGPersistArgs(ctx, &argv[argpos], argc - argpos,
                                  &(rinfo->dagTensorsPersistentContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        // TODO: clean temp structures
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "|>")) {
      // on the first pipe operator, if LOAD or PERSIST were used, we've already
      // allocated memory
      if (!((persistFlag == 1 || loadFlag == 1) && chainingOpCount == 0)) {
        rinfo->dagNumberCommands++;
        RAI_DagOp *currentDagOp = NULL;
        dagInit(&currentDagOp);
        array_append(rinfo->dagOps, currentDagOp);
      }
      chainingOpCount++;
    } else {
      if (!strcasecmp(arg_string, "AI.MODELRUN")) {
        if (argc - 2 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        RAI_Model *mto;
        RedisModuleKey *modelKey;
        const int status = RAI_GetModelFromKeyspace(ctx, argv[argpos+1], &modelKey,
                                                    &mto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          return REDISMODULE_ERR;
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->runkey = argv[argpos];
        // rinfo->mctx = RAI_ModelRunCtxCreate(mto);
        // This is the correct way but for now use the rinfo
        rinfo->dagOps[rinfo->dagNumberCommands]->mctx =
            RAI_ModelRunCtxCreate(mto);
      }
      RedisModule_RetainString(NULL, argv[argpos]);
      array_append(rinfo->dagOps[rinfo->dagNumberCommands]->argv, argv[argpos]);
      rinfo->dagOps[rinfo->dagNumberCommands]->argc++;
    }
  }

  // TODO: parse the entire command args and determine device
  RunQueueInfo *run_queue_info = NULL;
  // If the queue does not exist, initialize it
  if (ensureRunQueue("CPU",&run_queue_info) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(
        ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL,
                                          NULL, 0);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}

#define EXECUTION_PLAN_FREE_MSG 100

#define REGISTER_API(name, ctx) \
  if (RedisModule_ExportSharedAPI) {\
    if (RedisModule_ExportSharedAPI(ctx, "RedisAI_" #name, RAI_ ## name) != REDISMODULE_OK) {\
      RedisModule_Log(ctx, "warning", "Could not register RedisAI_%s", #name);\
      return REDISMODULE_ERR;\
    }\
  }

static int RAI_GetLLAPIVersion(){
  return REDISAI_LLAPI_VERSION;
}

static int RedisAI_RegisterApi(RedisModuleCtx* ctx) {

  if (!RedisModule_ExportSharedAPI) {
    RedisModule_Log(ctx, "warning", "Redis version does not support SharedAPI; running without exposing C API to other modules");
  }

  REGISTER_API(GetLLAPIVersion, ctx);

  REGISTER_API(TensorCreate, ctx);
  REGISTER_API(TensorDataSize, ctx);
  REGISTER_API(TensorFree, ctx);
  REGISTER_API(TensorSetData, ctx);
  REGISTER_API(TensorSetValueFromLongLong, ctx);
  REGISTER_API(TensorSetValueFromDouble, ctx);
  REGISTER_API(TensorGetValueAsDouble, ctx);
  REGISTER_API(TensorGetValueAsLongLong, ctx);
  REGISTER_API(TensorGetShallowCopy, ctx);
  REGISTER_API(TensorNumDims, ctx);
  REGISTER_API(TensorDim, ctx);
  REGISTER_API(TensorByteSize, ctx);
  REGISTER_API(TensorData, ctx);

  REGISTER_API(ModelCreate, ctx);
  REGISTER_API(ModelFree, ctx);
  REGISTER_API(ModelRunCtxCreate, ctx);
  REGISTER_API(ModelRunCtxAddInput, ctx);
  REGISTER_API(ModelRunCtxAddOutput, ctx);
  REGISTER_API(ModelRunCtxNumOutputs, ctx);
  REGISTER_API(ModelRunCtxOutputTensor, ctx);
  REGISTER_API(ModelRunCtxFree, ctx);
  REGISTER_API(ModelRun, ctx);
  REGISTER_API(ModelSerialize, ctx);
  REGISTER_API(ModelGetShallowCopy, ctx);

  REGISTER_API(ScriptCreate, ctx);
  REGISTER_API(ScriptFree, ctx);
  REGISTER_API(ScriptRunCtxCreate, ctx);
  REGISTER_API(ScriptRunCtxAddInput, ctx);
  REGISTER_API(ScriptRunCtxAddOutput, ctx);
  REGISTER_API(ScriptRunCtxNumOutputs, ctx);
  REGISTER_API(ScriptRunCtxOutputTensor, ctx);
  REGISTER_API(ScriptRunCtxFree, ctx);
  REGISTER_API(ScriptRun, ctx);
  REGISTER_API(ScriptGetShallowCopy, ctx);

  return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx, "ai", RAI_ENC_VER, REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  int flags = RedisModule_GetContextFlags(ctx);

  RAI_BackendsPath = NULL;

  if(RedisAI_RegisterApi(ctx) != REDISMODULE_OK){
    RedisModule_Log(ctx, "warning", "could not register RedisAI api\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_TensorInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize tensor dt\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_ModelInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize model dt\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_ScriptInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize script dt\r\n");
    return REDISMODULE_ERR;
  }

  if (RedisModule_CreateCommand(ctx, "ai.tensorset", RedisAI_TensorSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.tensorget", RedisAI_TensorGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelset", RedisAI_ModelSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelget", RedisAI_ModelGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modeldel", RedisAI_ModelDel_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelrun", RedisAI_ModelRun_RedisCommand, "write deny-oom getkeys-api", 3, 3, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai._modellist", RedisAI_ModelList_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptset", RedisAI_ScriptSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptget", RedisAI_ScriptGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptdel", RedisAI_ScriptDel_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptrun", RedisAI_ScriptRun_RedisCommand, "write deny-oom getkeys-api", 4, 4, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai._scriptlist", RedisAI_ScriptList_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.info", RedisAI_Info_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.config", RedisAI_Config_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.dagrun", RedisAI_DagRun_RedisCommand, "write deny-oom", 3, 3, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (argc > 0 && argc % 2 != 0) {
    RedisModule_Log(ctx, "warning", "Even number of arguments provided to module. Please provide arguments as KEY VAL pairs");
  }

  // need BACKENDSPATH set up before loading specific backends
  for (int i=0; i<argc/2; i++) {
    const char *key = RedisModule_StringPtrLen(argv[2*i], NULL);
    const char *val = RedisModule_StringPtrLen(argv[2*i + 1], NULL);

    int ret = REDISMODULE_OK;
    if (strcasecmp(key, "BACKENDSPATH") == 0) {
      ret = RedisAI_Config_BackendsPath(ctx, val);
    }
  }

  for (int i=0; i<argc/2; i++) {
    const char *key = RedisModule_StringPtrLen(argv[2*i], NULL);
    const char *val = RedisModule_StringPtrLen(argv[2*i + 1], NULL);

    int ret = REDISMODULE_OK;
    if (strcasecmp(key, "TF") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_TENSORFLOW, val);
    }
    else if (strcasecmp(key, "TFLITE") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_TFLITE, val);
    }
    else if (strcasecmp(key, "TORCH") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_TORCH, val);
    }
    else if (strcasecmp(key, "ONNX") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_ONNXRUNTIME, val);
    }
    // enable configuring the main thread to create a fixed number of worker threads up front per device.
    // by default we'll use 1
    else if (strcasecmp(key, "THREADS_PER_QUEUE") == 0) {
      ret = RedisAI_Config_QueueThreads(argv[2*i + 1]);
      if (ret == REDISMODULE_OK){
        char *buffer = RedisModule_Alloc((3 + strlen(REDISAI_INFOMSG_THREADS_PER_QUEUE) + strlen(val)) * sizeof(*buffer));
        sprintf(buffer, "%s: %s", REDISAI_INFOMSG_THREADS_PER_QUEUE, val);
        RedisModule_Log(ctx, "verbose", buffer);
        RedisModule_Free(buffer);
      }
    }
    else if (strcasecmp(key, "BACKENDSPATH") == 0) {
      // aleady taken care of
    } else {
      ret = REDISMODULE_ERR;
    }

    if (ret == REDISMODULE_ERR) {
      char* buffer = RedisModule_Alloc((4 + strlen(REDISAI_ERRORMSG_PROCESSING_ARG) + strlen(key) + strlen(val)) * sizeof(*buffer));
      sprintf(buffer, "%s: %s %s", REDISAI_ERRORMSG_PROCESSING_ARG, key, val);
      RedisModule_Log(ctx, "warning", buffer);
      RedisModule_Free(buffer);
    }
  }

  run_queues = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  RunQueueInfo *run_queue_info = NULL;
  if (ensureRunQueue("CPU",&run_queue_info) != REDISMODULE_OK){
    RedisModule_Log(ctx, "warning", "Queue not initialized for device CPU" );
    return REDISMODULE_ERR;
  }

  run_stats = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  
  return REDISMODULE_OK;
}

extern AI_dictType AI_dictTypeHeapStrings;
