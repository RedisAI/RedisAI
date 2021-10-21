#include "rai_rdb_decoder.h"
#include "current/v4/decode_v4.h"

void *RAI_RDBLoadTensor(RedisModuleIO *io) { return RAI_RDBLoadTensor_v4(io); }

void *RAI_RDBLoadModel(RedisModuleIO *io) { return RAI_RDBLoadModel_v4(io); }

void *RAI_RDBLoadScript(RedisModuleIO *io) { return RAI_RDBLoadScript_v4(io); }
