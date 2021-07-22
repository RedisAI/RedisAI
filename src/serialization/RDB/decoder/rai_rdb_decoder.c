#include "rai_rdb_decoder.h"
#include "current/v3/decode_v3.h"

void *RAI_RDBLoadTensor(RedisModuleIO *io) { return RAI_RDBLoadTensor_v3(io); }

void *RAI_RDBLoadModel(RedisModuleIO *io) { return RAI_RDBLoadModel_v3(io); }

void *RAI_RDBLoadScript(RedisModuleIO *io) { return RAI_RDBLoadScript_v3(io); }
