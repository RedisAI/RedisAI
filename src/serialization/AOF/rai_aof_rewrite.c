#include "rai_aof_rewrite.h"

void RAI_AOFRewriteTensor(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_Tensor *tensor = (RAI_Tensor *)value;

    char dtypestr[64];
    Tensor_DataTypeStr(RAI_TensorDataType(tensor), dtypestr);

    char *data = RAI_TensorData(tensor);
    long long size = RAI_TensorByteSize(tensor);

    long long ndims = RAI_TensorNumDims(tensor);

    RedisModuleString *dims[ndims];

    for (long long i = 0; i < ndims; i++) {
        dims[i] = RedisModule_CreateStringFromLongLong(RedisModule_GetContextFromIO(aof),
                                                       RAI_TensorDim(tensor, i));
    }

    RedisModule_EmitAOF(aof, "AI.TENSORSET", "scvcb", key, dtypestr, dims, ndims, "BLOB", data,
                        size);
}

void RAI_AOFRewriteModel(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_Model *model = (RAI_Model *)value;

    char *buffer = NULL;
    size_t len = 0;
    RAI_Error err = {0};

    int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

    if (err.code != RAI_OK) {

        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
        if (buffer) {
            RedisModule_Free(buffer);
        }
        return;
    }

    // AI.MODELSTORE model_key backend device [TAG tag]
    // [BATCHSIZE n [MINBATCHSIZE m [MINBATCHTIMEOUT t]]]
    // [INPUTS <input_count> name1 name2 ... OUTPUTS <output_count> name1 name2 ...]
    // BLOB model_blob

    long long chunk_size = Config_GetModelChunkSize();
    const size_t n_chunks = len / chunk_size + 1;
    RedisModuleString **buffers_ = array_new(RedisModuleString *, n_chunks);

    for (size_t i = 0; i < n_chunks; i++) {
        size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
        buffers_ = array_append(buffers_,
                                RedisModule_CreateString(NULL, buffer + i * chunk_size, chunk_len));
    }

    if (buffer) {
        RedisModule_Free(buffer);
    }

    const char *backendstr = RAI_GetBackendName(model->backend);

    if (model->backend != RAI_BACKEND_TENSORFLOW) {

        RedisModule_EmitAOF(aof, "AI.MODELSTORE", "scccsclclclcv", key, backendstr,
                            model->devicestr, "TAG", model->tag, "BATCHSIZE", model->opts.batchsize,
                            "MINBATCHSIZE", model->opts.minbatchsize, "MINBATCHTIMEOUT",
                            model->opts.minbatchtimeout, "BLOB", buffers_, n_chunks);
    } else {
        // For TF backend, the command should contain INPUTS and OUTPUTS names.
        // Create RedisModuleString* arrays from the char* arrays, so we can send a proper vector
        // to RedisModule_EmitAOF.
        array_new_on_stack(RedisModuleString *, 5, inputs_);
        array_new_on_stack(RedisModuleString *, 5, outputs_);

        for (size_t i = 0; i < model->ninputs; i++) {
            inputs_ = array_append(inputs_, RedisModule_CreateString(NULL, model->inputs[i],
                                                                     strlen(model->inputs[i])));
        }
        for (size_t i = 0; i < model->noutputs; i++) {
            outputs_ = array_append(outputs_, RedisModule_CreateString(NULL, model->outputs[i],
                                                                       strlen(model->outputs[i])));
        }

        RedisModule_EmitAOF(aof, "AI.MODELSTORE", "scccsclclclclvclvcv", key, backendstr,
                            model->devicestr, "TAG", model->tag, "BATCHSIZE", model->opts.batchsize,
                            "MINBATCHSIZE", model->opts.minbatchsize, "MINBATCHTIMEOUT",
                            model->opts.minbatchtimeout, "INPUTS", model->ninputs, inputs_,
                            model->ninputs, "OUTPUTS", model->noutputs, outputs_, model->noutputs,
                            "BLOB", buffers_, n_chunks);

        for (size_t i = 0; i < model->ninputs; i++) {
            RedisModule_FreeString(NULL, inputs_[i]);
        }
        array_free(inputs_);

        for (size_t i = 0; i < model->noutputs; i++) {
            RedisModule_FreeString(NULL, outputs_[i]);
        }
        array_free(outputs_);
    }

    for (size_t i = 0; i < n_chunks; i++) {
        RedisModule_FreeString(NULL, buffers_[i]);
    }
    array_free(buffers_);
}

void RAI_AOFRewriteScript(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_Script *script = (RAI_Script *)value;
    size_t nEntryPoints = array_len(script->entryPoints);
    array_new_on_stack(RedisModuleString*, nEntryPoints, entryPoints);
    for (size_t i =0; i< nEntryPoints; i++) {
        entryPoints = array_append(entryPoints, RedisModule_CreateString(NULL, script->entryPoints[i], strlen(script->entryPoints[i])));
    }
    RedisModule_EmitAOF(aof, "AI.SCRIPTSET", "sccsvlcc", key, script->devicestr, "TAG", script->tag, entryPoints, nEntryPoints,
                        "SOURCE", script->scriptdef);
    for(size_t i=0; i< nEntryPoints; i++) {
        RedisModule_FreeString(NULL, entryPoints[i]);
    }
    array_free(entryPoints);
}
