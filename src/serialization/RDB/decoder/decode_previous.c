#include "decode_previous.h"
#include "previous/v0/decode_v0.h"
#include "previous/v1/decode_v1.h"

void *Decode_PreviousTensor(RedisModuleIO *rdb, int encver) {
    switch (encver) {
    case 0:
        return RAI_RDBLoadTensor_v0(rdb);
    case 1:
        return RAI_RDBLoadTensor_v1(rdb);
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
    default:
        assert(false && "Invalid encoding version");
    }
    return NULL;
}