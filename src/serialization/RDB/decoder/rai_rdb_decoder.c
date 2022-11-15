/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "rai_rdb_decoder.h"
#include "current/v4/decode_v4.h"

void *RAI_RDBLoadTensor(RedisModuleIO *io) { return RAI_RDBLoadTensor_v4(io); }

void *RAI_RDBLoadModel(RedisModuleIO *io) { return RAI_RDBLoadModel_v4(io); }

void *RAI_RDBLoadScript(RedisModuleIO *io) { return RAI_RDBLoadScript_v4(io); }
