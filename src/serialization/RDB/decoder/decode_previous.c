/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "decode_previous.h"
#include "previous/v0/decode_v0.h"
#include "previous/v1/decode_v1.h"
#include "previous/v2/decode_v2.h"
#include "previous/v3/decode_v3.h"

void *Decode_PreviousTensor(RedisModuleIO *rdb, int encver) {
    switch (encver) {
    case 0:
        return RAI_RDBLoadTensor_v0(rdb);
    case 1:
        return RAI_RDBLoadTensor_v1(rdb);
    case 2:
        return RAI_RDBLoadTensor_v2(rdb);
    case 3:
        return RAI_RDBLoadTensor_v3(rdb);
    default:
        assert(false && "Invalid encoding version");
    }
    return NULL;
}

void *Decode_PreviousModel(RedisModuleIO *rdb, int encver) {
    switch (encver) {
    case 0:
        return RAI_RDBLoadModel_v0(rdb);
    case 1:
        return RAI_RDBLoadModel_v1(rdb);
    case 2:
        return RAI_RDBLoadModel_v2(rdb);
    case 3:
        return RAI_RDBLoadModel_v3(rdb);
    default:
        assert(false && "Invalid encoding version");
    }
    return NULL;
}

void *Decode_PreviousScript(RedisModuleIO *rdb, int encver) {
    switch (encver) {
    case 0:
        return RAI_RDBLoadScript_v0(rdb);
    case 1:
        return RAI_RDBLoadScript_v1(rdb);
    case 2:
        return RAI_RDBLoadScript_v2(rdb);
    case 3:
        return RAI_RDBLoadScript_v3(rdb);
    default:
        assert(false && "Invalid encoding version");
    }
    return NULL;
}