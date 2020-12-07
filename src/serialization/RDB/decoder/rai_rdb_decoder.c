#include "rai_rdb_decoder.h"
#include "current/v1/decode_v1.h"

void *RAI_RDBLoadTensor(RedisModuleIO *io) { return RAI_RDBLoadTensor_v1(io); }

void *RAI_RDBLoadModel(RedisModuleIO *io) { return RAI_RDBLoadModel_v1(io); }

void *RAI_RDBLoadScript(RedisModuleIO *io) { return RAI_RDBLoadScript_v1(io); }
