/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "config/config.h"
#include "tensor_struct.h"
#include "redis_ai_objects/stats.h"

typedef struct RAI_ModelOpts {
    size_t batchsize;
    size_t minbatchsize;
    size_t minbatchtimeout;
    long long backends_intra_op_parallelism; //  number of threads used within an
    //  individual op for parallelism.
    long long backends_inter_op_parallelism; //  number of threads used for parallelism
                                             //  between independent operations.
} RAI_ModelOpts;

typedef struct RAI_Model {
    void *model;
    // TODO: use session pool? The ideal would be to use one session per client.
    //       If a client disconnects, we dispose the session or reuse it for
    //       another client.
    void *session;
    RAI_Backend backend;
    char *devicestr;
    RedisModuleString *tag;
    RAI_ModelOpts opts;
    char **inputs;
    size_t ninputs;
    char **outputs;
    size_t noutputs;
    long long refCount;
    char *data;
    long long datalen;
    RAI_RunStats *info;
} RAI_Model;
