
#include "modelRun_ctx.h"

RAI_ModelRunCtx *RAI_ModelRunCtxCreate(RAI_Model *model) {
#define PARAM_INITIAL_SIZE 10
	RAI_ModelRunCtx *mctx = RedisModule_Calloc(1, sizeof(*mctx));
	mctx->model = RAI_ModelGetShallowCopy(model);
	mctx->inputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
	mctx->outputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
	return mctx;
#undef PARAM_INITIAL_SIZE
}

static int Model_RunCtxAddParam(RAI_ModelRunCtx *mctx, RAI_ModelCtxParam **paramArr,
  const char *name, RAI_Tensor *tensor) {

	RAI_ModelCtxParam param = {
	  .name = name,
	  .tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL,
	};
	*paramArr = array_append(*paramArr, param);
	return 1;
}

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx *mctx, const char *inputName, RAI_Tensor *inputTensor) {
	return Model_RunCtxAddParam(mctx, &mctx->inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx *mctx, const char *outputName) {
	return Model_RunCtxAddParam(mctx, &mctx->outputs, outputName, NULL);
}

size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx *mctx) { return array_len(mctx->inputs); }

size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx *mctx) { return array_len(mctx->outputs); }

RAI_Tensor *RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx *mctx, size_t index) {
	assert(RAI_ModelRunCtxNumInputs(mctx) > index && index >= 0);
	return mctx->inputs[index].tensor;
}

RAI_Tensor *RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx *mctx, size_t index) {
	assert(RAI_ModelRunCtxNumOutputs(mctx) > index && index >= 0);
	return mctx->outputs[index].tensor;
}

void RAI_ModelRunCtxFree(RAI_ModelRunCtx *mctx, int freeTensors) {
	if (freeTensors) {
		for (size_t i = 0; i < array_len(mctx->inputs); ++i) {
			RAI_TensorFree(mctx->inputs[i].tensor);
		}

		for (size_t i = 0; i < array_len(mctx->outputs); ++i) {
			if (mctx->outputs[i].tensor) {
				RAI_TensorFree(mctx->outputs[i].tensor);
			}
		}
	}

	array_free(mctx->inputs);
	array_free(mctx->outputs);

	RAI_Error err = {0};
	RAI_ModelFree(mctx->model, &err);

	if (err.code != RAI_OK) {
		// TODO: take it to client somehow
		RAI_ClearError(&err);
	}

	RedisModule_Free(mctx);
}

int RedisAI_Parse_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
  int argc, RAI_ModelRunCtx **mctx, RedisModuleString ***inkeys,
  RedisModuleString ***outkeys, RAI_Model **mto, RAI_Error *error) {
	if (argc < 3) {
		RAI_SetError(error, RAI_EMODELRUN,
		  "ERR wrong number of arguments for 'AI.MODELRUN' command");
		return -1;
	}

	const char *inputstr = RedisModule_StringPtrLen(argv[2], NULL);
	if (strcasecmp(inputstr, "INPUTS") != 0) {
		RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
		return -1;
	}

	int is_input = 0;
	size_t ninputs = 0;
	size_t noutputs = 0;
	int outputs_flag_count = 0;
	size_t argpos = 3;

	for (; argpos <= argc - 1; argpos++) {
		const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
		if (!strcasecmp(arg_string, "OUTPUTS") && outputs_flag_count == 0) {
			is_input = 1;
			outputs_flag_count = 1;
		} else {
			RedisModule_RetainString(ctx, argv[argpos]);
			if (is_input == 0) {
				*inkeys = array_append(*inkeys, argv[argpos]);
				ninputs++;
			} else {
				*outkeys = array_append(*outkeys, argv[argpos]);
				noutputs++;
			}
		}
	}
	if ((*mto)->inputs && array_len((*mto)->inputs) != ninputs) {
		RAI_SetError(error, RAI_EMODELRUN,
		  "Number of names given as INPUTS during MODELSET and keys given as "
		  "INPUTS here do not match");
		return -1;
	}

	if ((*mto)->outputs && array_len((*mto)->outputs) != noutputs) {
		RAI_SetError(error, RAI_EMODELRUN,
		  "Number of names given as OUTPUTS during MODELSET and keys given as "
		  "OUTPUTS here do not match");
		return -1;
	}
	return argpos;
}

int RedisAI_ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **argv,
  int argc, RAI_ModelRunCtx *mctx, RAI_Error *error, bool timeout) {

	RAI_Model *model = mctx->model;
	RAI_Tensor *t;
	RedisModuleKey *key;
	bool is_input = true;
	size_t input_pos = 0, output_pos = 0;
	// If timeout was given (it is non zero) inputs starts from position 5
	size_t arg_pos = timeout ? 5 : 3;
	for (; arg_pos < argc; arg_pos++) {
		RedisModuleString *keyName = argv[arg_pos];
		const char *arg_string = RedisModule_StringPtrLen(keyName, NULL);
		if (!strcasecmp(arg_string, "OUTPUTS")) {
			is_input = false;
			continue;
		}
		const char *opname = NULL;
		if (is_input) {
			const int status = RAI_GetTensorFromKeyspace(ctx, keyName, &key, &t,
			  REDISMODULE_READ);
			if (status == REDISMODULE_ERR) {
				RedisModule_Log(ctx, "warning",
				  "could not load tensor %s from keyspace", arg_string);
				return REDISMODULE_ERR;
			}
			RedisModule_CloseKey(key);
			if (model->inputs)
				opname = model->inputs[input_pos++];
			RAI_ModelRunCtxAddInput(mctx, opname, t);

		} else {
			if (model->outputs)
				opname = model->outputs[output_pos++];
			RAI_ModelRunCtxAddOutput(mctx, opname);
		}
	}
	return REDISMODULE_OK;
}


RedisAI_RunInfo *Dag_CreateFromSingleModelRunOp(RAI_ModelRunCtx *mctx, RAI_Error *error,
  RedisModuleString **inkeys, RedisModuleString **outkeys,
  RedisModuleString *runkey, long long timeout) {
	RedisAI_RunInfo *rinfo = NULL;
	if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
		RAI_SetError(error, RedisAI_ErrorCode_EMODELRUN,
		  "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
		return NULL;
	}
	rinfo->single_device_dag = 1;
	rinfo->single_op_dag = 1;
	rinfo->dagOpCount = 1;
	rinfo->timeout = timeout;

	RAI_DagOp *currentDagOp;
	if (RAI_InitDagOp(&currentDagOp) == REDISMODULE_ERR) {
		RAI_SetError(error, RedisAI_ErrorCode_EMODELRUN,
		  "ERR Unable to allocate the memory and initialise the RAI_dagOp structure");
		return NULL;
	};
	rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);

	RAI_DagOp *currentOp = rinfo->dagOps[0];
	currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
	currentOp->mctx = mctx;
	currentOp->devicestr = mctx->model->devicestr;
	currentOp->inkeys = inkeys;
	currentOp->outkeys = outkeys;
	currentOp->runkey = runkey;

	return rinfo;
}
