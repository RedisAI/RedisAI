#include "rai_rdb_decoder.h"
#include "current/v2/decode_v2.h"

void *RAI_RDBLoadTensor(RedisModuleIO *io) { return RAI_RDBLoadTensor_v2(io); }

void *RAI_RDBLoadModel(RedisModuleIO *io) { return RAI_RDBLoadModel_v2(io); }

void *RAI_RDBLoadScript(RedisModuleIO *io) { return RAI_RDBLoadScript_v2(io); }
