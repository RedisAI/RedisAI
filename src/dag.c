/**
 * dag.c
 *
 * Contains the helper methods for both parsing, running the command in the
 * background, and replying DAG structured commands.
 * 
 * The way we allow DAG operations to run on different devices in parallel
 * (when possible) is the following: instead of running the whole DAG in one
 * swoop, the DAG run info is created on one
 * queue/device and shallow copied (appropriately) across other queues/devices
 * as indicated by the DAG specification. A DAG mutex is shared across all
 * copies.
 * The DAG run info is placed on the queue for each device and evicted for
 * execution (in background_workers). Execution happens one DAG op at a time:
 * once the individual op has executed, it is marked as such and the DAG run
 * info is placed back on the queue. The current un-executed op is checked for
 * its inputs. If all inputs are found in the tensor context, then the DAG op
 * can be executed. If not, the execution quits and control is given back to
 * the worker. If there are other items in the queue the op is placed after the
 * next item. When all ops for a device have been executed, the DAG is not
 * placed back on the queue. When all ops in a DAG have been executed or an
 * error occurs, the client is unblocked.
 * 
 * See background_workers.c for the queue logic, everything else DAG is here.
 */

#include "dag.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "model.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

/**
 * Execution of a TENSORSET DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp TENSORSET DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_TensorSet_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
  RAI_Tensor *t = NULL;
  const int parse_result = RAI_parseTensorSetArgs(
      NULL, currentOp->argv, currentOp->argc, &t, 0, currentOp->err);
  if (parse_result > 0) {
    const char *key_string =
        RedisModule_StringPtrLen(currentOp->outkeys[0], NULL);
    pthread_mutex_lock(rinfo->dagMutex);
    AI_dictReplace(rinfo->dagTensorsContext, (void*)key_string, t);
    pthread_mutex_unlock(rinfo->dagMutex);
    currentOp->result = REDISMODULE_OK;
  } else {
    currentOp->result = REDISMODULE_ERR;
  }
}

/**
 * Execution of a TENSORGET DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp TENSORGET DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_TensorGet_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
  const char *key_string = RedisModule_StringPtrLen(currentOp->inkeys[0], NULL);
  RAI_Tensor *t = NULL;
  pthread_mutex_lock(rinfo->dagMutex);
  currentOp->result = RAI_getTensorFromLocalContext(
      NULL, rinfo->dagTensorsContext, key_string, &t, currentOp->err);
  pthread_mutex_unlock(rinfo->dagMutex);
  if (currentOp->result == REDISMODULE_OK) {
    RAI_Tensor *outTensor = NULL;
    // TODO: check tensor copy return value
    RAI_TensorDeepCopy(t, &outTensor);
    currentOp->outTensors = array_append(currentOp->outTensors, outTensor);
  }
}

/**
 * Execution of a MODELRUN DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp MODELRUN DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_ModelRun_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
  pthread_mutex_lock(rinfo->dagMutex);

  uint n_inkeys = array_len(currentOp->inkeys);
  uint n_outkeys = array_len(currentOp->outkeys);

  RAI_Tensor* inputTensors[n_inkeys];
  for (uint i=0; i<n_inkeys; i++) {
    RAI_Tensor *inputTensor;
    const int get_result = RAI_getTensorFromLocalContext(
        NULL, rinfo->dagTensorsContext, RedisModule_StringPtrLen(currentOp->inkeys[i], NULL), &inputTensor, currentOp->err);
    if (get_result == REDISMODULE_ERR) {
      // We check for this outside the function
      // this check cannot be covered by tests
      currentOp->result = REDISMODULE_ERR;
      pthread_mutex_unlock(rinfo->dagMutex);
      return;
    }
    inputTensors[i] = inputTensor;
  }

  for (uint i=0; i<n_inkeys; i++) {
    const char *opname = NULL;
    if (currentOp->mctx->model->inputs) {
      opname = currentOp->mctx->model->inputs[i];
    }
    RAI_ModelRunCtxAddInput(currentOp->mctx, opname, inputTensors[i]);
  }

  for (uint i=0; i<n_outkeys; i++) {
    const char *opname = NULL;
    if (currentOp->mctx->model->inputs) {
      opname = currentOp->mctx->model->outputs[i];
    }
    RAI_ModelRunCtxAddOutput(currentOp->mctx, opname);
  }

  pthread_mutex_unlock(rinfo->dagMutex);

  RAI_ModelRunCtx *mctxs[1];
  mctxs[0] = currentOp->mctx;
  const long long start = ustime();
  int result = RAI_ModelRun(mctxs, 1, currentOp->err);
  const long long end = ustime();

  pthread_mutex_lock(rinfo->dagMutex);

  if (result == REDISMODULE_ERR) {
    currentOp->result = result;
    pthread_mutex_unlock(rinfo->dagMutex);
    return;
  }

  currentOp->duration_us = end - start;

  const size_t noutputs = RAI_ModelRunCtxNumOutputs(currentOp->mctx);
  for (size_t outputNumber = 0; outputNumber<noutputs; outputNumber++) {
    RAI_Tensor *tensor = RAI_ModelRunCtxOutputTensor(currentOp->mctx, outputNumber);
    const char *key_string = RedisModule_StringPtrLen(
        currentOp->outkeys[outputNumber], NULL);
    AI_dictReplace(rinfo->dagTensorsContext, (void*)key_string, tensor ? RAI_TensorGetShallowCopy(tensor) : NULL);
  }

  currentOp->result = result;

  pthread_mutex_unlock(rinfo->dagMutex);

  return;
}

/**
 * Execution of a SCRIPTRUN DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp SCRIPTRUN DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_ScriptRun_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
  pthread_mutex_lock(rinfo->dagMutex);

  uint n_inkeys = array_len(currentOp->inkeys);
  uint n_outkeys = array_len(currentOp->outkeys);

  RAI_Tensor* inputTensors[n_inkeys];
  for (uint i=0; i<n_inkeys; i++) {
    RAI_Tensor *inputTensor;
    const int get_result = RAI_getTensorFromLocalContext(
        NULL, rinfo->dagTensorsContext, RedisModule_StringPtrLen(currentOp->inkeys[i], NULL), &inputTensor, currentOp->err);
    if (get_result == REDISMODULE_ERR) {
      // We check for this outside the function
      // this check cannot be covered by tests
      currentOp->result = REDISMODULE_ERR;
      pthread_mutex_unlock(rinfo->dagMutex);
      return;
    }
    inputTensors[i] = inputTensor;
  }

  for (uint i=0; i<n_inkeys; i++) {
    RAI_ScriptRunCtxAddInput(currentOp->sctx, inputTensors[i], currentOp->err);
  }

  for (uint i=0; i<n_outkeys; i++) {
    RAI_ScriptRunCtxAddOutput(currentOp->sctx);
  } 

  pthread_mutex_unlock(rinfo->dagMutex);

  const long long start = ustime();
  int result = RAI_ScriptRun(currentOp->sctx, currentOp->err);
  const long long end = ustime();

  pthread_mutex_lock(rinfo->dagMutex);

  const size_t noutputs = RAI_ScriptRunCtxNumOutputs(currentOp->sctx);
  for (size_t outputNumber = 0; outputNumber < noutputs;
     outputNumber++) {
    RAI_Tensor *tensor =
          RAI_ScriptRunCtxOutputTensor(currentOp->sctx, outputNumber);
    const char *key_string = RedisModule_StringPtrLen(
            currentOp->outkeys[outputNumber], NULL);
    AI_dictReplace(rinfo->dagTensorsContext, (void*)key_string, tensor ? RAI_TensorGetShallowCopy(tensor) : NULL);
  }

  currentOp->result = result;
  currentOp->duration_us = end - start;

  pthread_mutex_unlock(rinfo->dagMutex);

  return;
}

void RedisAI_DagRunSessionStep(RedisAI_RunInfo *rinfo, const char *devicestr,
                               int *progress,
                               int *device_complete, int *all_devices_complete) {
  RAI_DagOp *currentOp = NULL;

  pthread_mutex_lock(rinfo->dagMutex);

  *rinfo->dagRefCount += 1;

  int all_devices_complete_ = 1;
  int device_complete_ = 1;
  for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {

    if (rinfo->dagOps[i]->result >= 0) {
      continue;
    }

    all_devices_complete_ = 0;

    if (strcasecmp(devicestr, rinfo->dagOps[i]->devicestr) == 0) {
      device_complete_ = 0;
      currentOp = rinfo->dagOps[i];
      break;
    }
  }

  *device_complete = device_complete_;
  *all_devices_complete = all_devices_complete_;
 
  if (currentOp == NULL && device_complete) {
    *rinfo->dagRefCount -= 1;
    pthread_mutex_unlock(rinfo->dagMutex);
    return;
  }

  uint n_inkeys = array_len(currentOp->inkeys);
  for (int i=0; i<n_inkeys; i++) {
    if (AI_dictFind(rinfo->dagTensorsContext,
                    RedisModule_StringPtrLen(currentOp->inkeys[i], NULL)) == NULL) {
      *rinfo->dagRefCount -= 1;
      pthread_mutex_unlock(rinfo->dagMutex);
      *progress = 0;
      return;
    }
  }

  pthread_mutex_unlock(rinfo->dagMutex);

  switch (currentOp->commandType) {
    case REDISAI_DAG_CMD_TENSORSET: {
      RedisAI_DagRunSession_TensorSet_Step(rinfo, currentOp);
      break;
    }
    case REDISAI_DAG_CMD_TENSORGET: {
      RedisAI_DagRunSession_TensorGet_Step(rinfo, currentOp);
      break;
    }
    case REDISAI_DAG_CMD_MODELRUN: {
      RedisAI_DagRunSession_ModelRun_Step(rinfo, currentOp);
      break;
    }
    case REDISAI_DAG_CMD_SCRIPTRUN: {
      RedisAI_DagRunSession_ScriptRun_Step(rinfo, currentOp);
      break;
    }
    default: {
      /* unsupported DAG's command */
      RAI_SetError(currentOp->err, RAI_EDAGRUN, "ERR unsupported command within DAG");
      currentOp->result = REDISMODULE_ERR;
      break;
    }
  }

  pthread_mutex_lock(rinfo->dagMutex);

  if (currentOp->result == REDISMODULE_OK) {
    *progress = 1;
  }
  else {
    *progress = 0;
    *rinfo->dagError = 1;
  }

  *rinfo->dagRefCount -= 1;

  pthread_mutex_unlock(rinfo->dagMutex);
  
  return;
}

int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);

  int dag_error = 0;
  char *detail_oneline;

  size_t n_dagOps = array_len(rinfo->dagOps);

  RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_ARRAY_LEN);
  for (size_t i = 0; i < n_dagOps; i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];
    switch (currentOp->commandType) {
      case REDISAI_DAG_CMD_TENSORSET: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
          dag_error = 1;
        }
        else if (currentOp->result == -1) {
          RedisModule_ReplyWithSimpleString(ctx, "NA");
        }
        else {
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

      case REDISAI_DAG_CMD_TENSORGET: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
          dag_error = 1;
        } else {
          if (array_len(currentOp->outTensors) > 0) {
            RAI_Tensor *tensor = currentOp->outTensors[0];
            RAI_parseTensorGetArgs(ctx, currentOp->argv, currentOp->argc,
                                   tensor);
          }
          else if (currentOp->result == -1) {
            RedisModule_ReplyWithSimpleString(ctx, "NA");
          }
          else {
            RedisModule_ReplyWithError(
                ctx, "ERR error getting tensor from local context");
          }
        }
        break;
      }

      case REDISAI_DAG_CMD_MODELRUN: {
        rinfo->dagReplyLength++;
        struct RedisAI_RunStats *rstats = NULL;
        const char *runkey =
            RedisModule_StringPtrLen(currentOp->runkey, NULL);
        RAI_GetRunStats(runkey,&rstats);
        if (currentOp->result == REDISMODULE_ERR) {
          RAI_SafeAddDataPoint(rstats,0,1,1,0);
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
          dag_error = 1;
        }
        else if (currentOp->result == -1) {
          RedisModule_ReplyWithSimpleString(ctx, "NA");
        }
        else {
          RAI_SafeAddDataPoint(rstats,currentOp->duration_us,1,0,0);
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

      case REDISAI_DAG_CMD_SCRIPTRUN: {
        rinfo->dagReplyLength++;
        struct RedisAI_RunStats *rstats = NULL;
        const char *runkey = RedisModule_StringPtrLen(currentOp->runkey, NULL);
        RAI_GetRunStats(runkey,&rstats);
        if (currentOp->result == REDISMODULE_ERR) {
          RAI_SafeAddDataPoint(rstats,0,1,1,0);
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
          dag_error = 1;
        }
        else if (currentOp->result == -1) {
          RedisModule_ReplyWithSimpleString(ctx, "NA");
        }
        else {
          RAI_SafeAddDataPoint(rstats,currentOp->duration_us,1,0,0);
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

      default:
        /* no-op */
        break;
    }
  }

  if (dag_error) {
    RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
    RAI_FreeRunInfo(ctx, rinfo);
    return REDISMODULE_ERR;
  }

  AI_dictIterator *persist_iter =
      AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
  AI_dictEntry *persist_entry = AI_dictNext(persist_iter);
  while (persist_entry) {
    const char *persist_key_name = AI_dictGetKey(persist_entry);
    AI_dictEntry *tensor_entry =
        AI_dictFind(rinfo->dagTensorsContext, persist_key_name);
    if (tensor_entry) {
      RAI_Tensor *tensor = RAI_TensorGetShallowCopy(AI_dictGetVal(tensor_entry));
      RedisModuleKey *key;
      char *demangled_key_name = RedisModule_Strdup(persist_key_name);
      demangled_key_name[strlen(persist_key_name) - 4] = 0;
      RedisModuleString *tensor_keyname = RedisModule_CreateString(
          ctx, demangled_key_name, strlen(demangled_key_name));
      const int status = RAI_OpenKey_Tensor(
          ctx, tensor_keyname, &key, REDISMODULE_READ | REDISMODULE_WRITE);
      RedisModule_Free(demangled_key_name);
      if (status == REDISMODULE_ERR) {
        RAI_TensorFree(tensor);
        RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
        rinfo->dagReplyLength++;
      } else {
        if (RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, tensor) !=
            REDISMODULE_OK) {
          RAI_TensorFree(tensor);
          RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
          rinfo->dagReplyLength++;
        }
      }
      RedisModule_CloseKey(key);
      RedisAI_ReplicateTensorSet(ctx, tensor_keyname, tensor);
    } else {
      RedisModule_ReplyWithError(
          ctx, "ERR specified persistent key that was not used in DAG");
      rinfo->dagReplyLength++;

      RedisModule_Log(ctx, "warning",
                      "on DAGRUN's PERSIST pecified persistent key (%s) that "
                      "was not used on DAG. Logging all local context keys",
                      persist_key_name);
      AI_dictIterator *local_iter =
          AI_dictGetSafeIterator(rinfo->dagTensorsContext);
      AI_dictEntry *local_entry = AI_dictNext(local_iter);
      while (local_entry) {
        const char *localcontext_key_name = AI_dictGetKey(local_entry);
        RedisModule_Log(ctx, "warning", "DAG's local context key (%s)",
                        localcontext_key_name);
        local_entry = AI_dictNext(local_iter);
      }

      for (size_t opN = 0; opN < array_len(rinfo->dagOps); opN++) {
        RedisModule_Log(
            ctx, "warning", "DAG's op n#  %zu - cmdType %d ( argc %d )", opN,
            rinfo->dagOps[opN]->commandType, rinfo->dagOps[opN]->argc);
      }
    }

    persist_entry = AI_dictNext(persist_iter);
  }

  AI_dictReleaseIterator(persist_iter);
  RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
  RAI_FreeRunInfo(ctx, rinfo);

  return REDISMODULE_OK;
}

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 */
int RAI_parseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc, AI_dict **loadedContextDict,
                         AI_dict **localContextDict,
                         const char *chaining_operator) {
  if (argc < 3) {
    RedisModule_WrongArity(ctx);
    return -1;
  }

  long long n_keys;
  const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
  if (retval != REDISMODULE_OK || n_keys <= 0) {
    RedisModule_ReplyWithError(
        ctx, "ERR invalid or negative value found in number of keys to LOAD");
    return -1;
  }
  int number_loaded_keys = 0;
  int separator_flag = 0;
  size_t argpos = 2;
  for (; (argpos <= argc - 1) && (number_loaded_keys < n_keys); argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, chaining_operator)) {
      separator_flag = 1;
      break;
    } else {
      RAI_Tensor *t;
      RedisModuleKey *key;
      const int status = RAI_GetTensorFromKeyspace(ctx, argv[argpos], &key, &t, REDISMODULE_READ);
      if (status == REDISMODULE_ERR) {
        RedisModule_Log(
            ctx, "warning",
            "on DAGRUN's LOAD could not load tensor %s from keyspace",
            arg_string);
        return -1;
      }
      RedisModule_CloseKey(key);
      char *dictKey = (char*) RedisModule_Alloc((strlen(arg_string) + 5)*sizeof(char));
      sprintf(dictKey, "%s%04d", arg_string, 1);
      AI_dictAdd(*localContextDict, (void*)dictKey, (void *)RAI_TensorGetShallowCopy(t));
      AI_dictAdd(*loadedContextDict, (void*)dictKey, (void *)1);
      RedisModule_Free(dictKey);
      number_loaded_keys++;
    }
  }
  if (number_loaded_keys != n_keys) {
    RedisModule_WrongArity(ctx);
    return -1;
  }
  return argpos;
}

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 */
int RAI_parseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                            int argc, AI_dict **persistContextDict,
                            const char *chaining_operator) {
  if (argc < 3) {
    RedisModule_WrongArity(ctx);
    return -1;
  }

  long long n_keys;
  const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
  if (retval != REDISMODULE_OK || n_keys <= 0) {
    RedisModule_ReplyWithError(
        ctx,
        "ERR invalid or negative value found in number of keys to PERSIST");
    return -1;
  }

  int number_loaded_keys = 0;
  int separator_flag = 0;
  size_t argpos = 2;
  for (; (argpos < argc) && (number_loaded_keys < n_keys); argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, chaining_operator)) {
      separator_flag = 1;
      break;
    } else {
      AI_dictAdd(*persistContextDict, (void*)arg_string, (void *)1);
      number_loaded_keys++;
    }
  }
  if (number_loaded_keys != n_keys) {
    RedisModule_WrongArity(ctx);
    return -1;
  }
  return argpos;
}

int RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                    RedisModuleString **argv, int argc){
    for (size_t argpos = 1; argpos < argc; argpos++){
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if ( (!strcasecmp(arg_string, "LOAD") || !strcasecmp(arg_string, "PERSIST") ) && (argpos+1 < argc) ) {
            long long n_keys;
            argpos++;
            const int retval = RedisModule_StringToLongLong(argv[argpos], &n_keys);
            if(retval != REDISMODULE_OK){
                return REDISMODULE_ERR;
            }
            argpos++;
            if (n_keys > 0){
                size_t last_persist_argpos = n_keys+argpos;
                for (; argpos < last_persist_argpos &&  argpos < argc; argpos++){
                    RedisModule_KeyAtPos(ctx, argpos);
                }
            }
        }
    }
    return REDISMODULE_OK;
}

int RedisAI_DagRunSyntaxParser(RedisModuleCtx *ctx, RedisModuleString **argv,
                                 int argc, int dagMode) {
  if (RedisModule_IsKeysPositionRequest(ctx)) {
     return RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
  }
  if (argc < 4) return RedisModule_WrongArity(ctx);
  RedisAI_RunInfo *rinfo = NULL;
  if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(
        ctx,
        "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo "
        "structure");
  }
  rinfo->use_local_context = 1;
  RAI_DagOp *currentDagOp = NULL;
  RAI_InitDagOp(&currentDagOp);
  rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);

  int persistFlag = 0;
  int loadFlag = 0;
  int chainingOpCount = 0;

  for (size_t argpos = 1; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "LOAD")) {
      loadFlag = 1;
      const int parse_result = RAI_parseDAGLoadArgs(
          ctx, &argv[argpos], argc - argpos, &(rinfo->dagTensorsLoadedContext),
          &(rinfo->dagTensorsContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx, rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "PERSIST")) {
      if (dagMode == REDISAI_DAG_READONLY_MODE) {
        RAI_FreeRunInfo(ctx, rinfo);
        return RedisModule_ReplyWithError(
            ctx, "ERR PERSIST cannot be specified in a read-only DAG");
      }
      persistFlag = 1;
      const int parse_result =
          RAI_parseDAGPersistArgs(ctx, &argv[argpos], argc - argpos,
                                  &(rinfo->dagTensorsPersistedContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx, rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "|>")) {
      // on the first pipe operator, if LOAD or PERSIST were used, we've already
      // allocated memory
      if (chainingOpCount > 0) {
        rinfo->dagNumberCommands++;
        RAI_DagOp *currentDagOp = NULL;
        RAI_InitDagOp(&currentDagOp);
        rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);
      }
      chainingOpCount++;
    } else {
      if (!strcasecmp(arg_string, "AI.TENSORGET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType =
            REDISAI_DAG_CMD_TENSORGET;
        rinfo->dagOps[rinfo->dagNumberCommands]->devicestr = "CPU";
      }
      if (!strcasecmp(arg_string, "AI.TENSORSET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType =
            REDISAI_DAG_CMD_TENSORSET;
        rinfo->dagOps[rinfo->dagNumberCommands]->devicestr = "CPU";
      }
      if (!strcasecmp(arg_string, "AI.MODELRUN")) {
        if (argc - 2 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagNumberCommands];
        currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
        RAI_Model *mto;
        RedisModuleKey *modelKey;
        const int status = RAI_GetModelFromKeyspace(
            ctx, argv[argpos + 1], &modelKey, &mto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          RAI_FreeRunInfo(ctx, rinfo);
          return REDISMODULE_ERR;
        }
        currentOp->devicestr = mto->devicestr;
        currentOp->runkey = argv[argpos + 1];
        currentOp->mctx = RAI_ModelRunCtxCreate(mto);
      }
      if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
        if (argc - 3 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagNumberCommands];
        currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
        RAI_Script *sto;
        RedisModuleKey *scriptKey;
        const int status = RAI_GetScriptFromKeyspace(
            ctx, argv[argpos + 1], &scriptKey, &sto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          RAI_FreeRunInfo(ctx, rinfo);
          return REDISMODULE_ERR;
        }
        currentOp->devicestr = sto->devicestr;
        const char *functionName =
            RedisModule_StringPtrLen(argv[argpos + 2], NULL);
        currentOp->runkey = argv[argpos + 1];
        currentOp->sctx = RAI_ScriptRunCtxCreate(sto, functionName);
      }
      RedisModule_RetainString(NULL, argv[argpos]);
      rinfo->dagOps[rinfo->dagNumberCommands]->argv = array_append(rinfo->dagOps[rinfo->dagNumberCommands]->argv, argv[argpos]);
      rinfo->dagOps[rinfo->dagNumberCommands]->argc++;
    }
  }

  for (long long i=0; i<array_len(rinfo->dagOps); i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];
    if(currentOp==NULL) continue;
    int parse_result;
    switch (currentOp->commandType) {
      case REDISAI_DAG_CMD_TENSORSET:
        currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
        break;
      case REDISAI_DAG_CMD_TENSORGET:
        currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
        break;
      case REDISAI_DAG_CMD_MODELRUN:
        parse_result = RedisAI_Parse_ModelRun_RedisCommand(
            NULL, currentOp->argv, currentOp->argc, &(currentOp->mctx),
            &(currentOp->inkeys), &(currentOp->outkeys),
            &(currentOp->mctx->model), currentOp->err);
        if (parse_result < 0) {
          return RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        }
        break;
      case REDISAI_DAG_CMD_SCRIPTRUN:
        parse_result = RedisAI_Parse_ScriptRun_RedisCommand(
                NULL, currentOp->argv, currentOp->argc,
                &(currentOp->inkeys), &(currentOp->outkeys),
                &(currentOp->sctx->variadic), currentOp->err);
        if (parse_result < 0) {
          return RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        }
        break;
    }
  }

  // At this point, we have built a sequence of DAG operations, each with its own
  // input and output keys. The names of the keys will be used to look whether the
  // inputs to a DAG operation have all been realized by previous operations (or if
  // they are available as part of LOADed keys from keyspace).
  // This strategy is fine if keys are not aliased, that is, if a command's output
  // overwrites the key of a previous command. This would trick DAG operations into
  // thinking that their input is ready when it's not.
  // To overcome this, we make key names unique, so that names are not aliased. We
  // mangle the names by appending a numerical suffix ":0001". After computing, we
  // demangle the keys in order to persist them.

  AI_dict* mangled_tensors = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  if (!mangled_tensors) {
    return REDISMODULE_ERR;
  }

  {
    AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsLoadedContext);
    AI_dictEntry *entry = AI_dictNext(iter);
    while (entry) {
      char *key = (char *)AI_dictGetKey(entry);
      char *demangled_key = RedisModule_Strdup(key);
      demangled_key[strlen(key) - 4] = 0;
      int *instance = RedisModule_Alloc(sizeof(int));
      *instance = 1;
      AI_dictAdd(mangled_tensors, (void *)demangled_key, (void *)instance);
      RedisModule_Free(demangled_key);
      entry = AI_dictNext(iter);
    }
    AI_dictReleaseIterator(iter);
  }

  for (long long i=0; i<array_len(rinfo->dagOps); i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];

    RedisModuleString **mangled_inkeys = array_new(RedisModuleString*, array_len(currentOp->inkeys));
    for (long long j=0; j<array_len(currentOp->inkeys); j++) {
      const char* key = RedisModule_StringPtrLen(currentOp->inkeys[j], NULL);
      AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
      if (!entry) {
        return RedisModule_ReplyWithError(ctx,
                                          "ERR INPUT key cannot be found in DAG");
      }
      int *instance = AI_dictGetVal(entry);
      RedisModuleString *mangled_key = RedisModule_CreateStringPrintf(ctx, "%s%04d", key, *instance);
      mangled_inkeys = array_append(mangled_inkeys, mangled_key);
    }

    RedisModuleString **mangled_outkeys = array_new(RedisModuleString*, array_len(currentOp->outkeys));
    for (long long j=0; j<array_len(currentOp->outkeys); j++) {
      const char* key = RedisModule_StringPtrLen(currentOp->outkeys[j], NULL);
      AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
      int *instance = NULL;
      if (entry) {
        instance = AI_dictGetVal(entry);
        *instance += 1;
      }
      else {
        instance = RedisModule_Alloc(sizeof(int));
        *instance = 1;
        AI_dictAdd(mangled_tensors, (void *)key, (void *)instance);
      }
      RedisModuleString *mangled_key = RedisModule_CreateStringPrintf(ctx, "%s%04d", key, *instance);
      mangled_outkeys = array_append(mangled_outkeys, mangled_key);
    }
  
    array_free(currentOp->inkeys);
    array_free(currentOp->outkeys);

    currentOp->inkeys = mangled_inkeys;
    currentOp->outkeys = mangled_outkeys;
  }

  AI_dict* mangled_persisted = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  {
    AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
    AI_dictEntry *entry = AI_dictNext(iter);
    while (entry) {
      char *key = (char *)AI_dictGetKey(entry);
      AI_dictEntry *mangled_entry = AI_dictFind(mangled_tensors, key);
      if (!mangled_entry) {
        return RedisModule_ReplyWithError(ctx,
                                          "ERR PERSIST key cannot be found in DAG");
      } 
      int *instance = AI_dictGetVal(mangled_entry);
      RedisModuleString *mangled_key = RedisModule_CreateStringPrintf(ctx, "%s%04d", key, *instance);
      const char* mangled_key_str = RedisModule_StringPtrLen(mangled_key, NULL);
      AI_dictAdd(mangled_persisted, (void *)mangled_key_str, (void *)1);
      entry = AI_dictNext(iter);
    }
    AI_dictReleaseIterator(iter);
  }

  rinfo->dagTensorsPersistedContext = mangled_persisted;

  {
    AI_dictIterator *iter = AI_dictGetSafeIterator(mangled_tensors);
    AI_dictEntry *entry = AI_dictNext(iter);
    while (entry) {
      int *val = (int *)AI_dictGetVal(entry);
      RedisModule_Free(val);
      entry = AI_dictNext(iter);
    }
    AI_dictReleaseIterator(iter);
  }
  AI_dictRelease(mangled_tensors);
  mangled_tensors = NULL;

  for (long long i=0; i<array_len(rinfo->dagOps); i++) {
    if (rinfo->dagOps[i]->devicestr == NULL) {
      rinfo->dagOps[i]->devicestr = "CPU";
    }
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, NULL, 0);

  const char **devices = array_new(const char *, 10);

  for (long long i=0; i<array_len(rinfo->dagOps); i++) {
    const char* devicestr = rinfo->dagOps[i]->devicestr;
    bool found = false;
    for (long long j=0; j<array_len(devices); j++) {
      if (strcasecmp(devicestr, devices[j]) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      devices = array_append(devices, devicestr);
    }
  }
  
  const char* master_device = rinfo->dagOps[array_len(rinfo->dagOps)-1]->devicestr;

  size_t ndevices = array_len(devices);

  *rinfo->dagRefCount = 0;

  RedisAI_RunInfo **rinfo_copies = array_new(RedisAI_RunInfo*, ndevices - 1);
  
  for (long long i=0; i<ndevices-1; i++) {
    RedisAI_RunInfo *rinfo_copy;
    RAI_ShallowCopyDagRunInfo(&rinfo_copy, rinfo);
    rinfo_copies = array_append(rinfo_copies, rinfo_copy);
  }

  int copy_count = 0;
  for (long long i=0; i<ndevices; i++) {
    const char* devicestr = devices[i];
    RunQueueInfo *run_queue_info = NULL;
    if (ensureRunQueue(devicestr, &run_queue_info) == REDISMODULE_ERR) {
      RAI_FreeRunInfo(ctx, rinfo);
      return RedisModule_ReplyWithError(ctx,
                                        "ERR Queue not initialized for device");
    }

    RedisAI_RunInfo *curr_rinfo = NULL;

    if (!strcasecmp(devicestr, master_device)) {
      curr_rinfo = rinfo;
    }
    else {
      curr_rinfo = rinfo_copies[copy_count];
      copy_count += 1;
    }

    pthread_mutex_lock(&run_queue_info->run_queue_mutex);
    queuePush(run_queue_info->run_queue, curr_rinfo);
    pthread_cond_signal(&run_queue_info->queue_condition_var);
    pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
  }

  array_free(devices);
  array_free(rinfo_copies);

  return REDISMODULE_OK;
}
