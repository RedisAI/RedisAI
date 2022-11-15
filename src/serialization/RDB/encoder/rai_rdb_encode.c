/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "rai_rdb_encode.h"
#include "v4/encode_v4.h"

void RAI_RDBSaveTensor(RedisModuleIO *io, void *value) { RAI_RDBSaveTensor_v4(io, value); }

void RAI_RDBSaveModel(RedisModuleIO *io, void *value) { RAI_RDBSaveModel_v4(io, value); }

void RAI_RDBSaveScript(RedisModuleIO *io, void *value) { RAI_RDBSaveScript_v4(io, value); }
