#include "rai_rdb_encode.h"
#include "v2/encode_v2.h"

void RAI_RDBSaveTensor(RedisModuleIO *io, void *value) { RAI_RDBSaveTensor_v2(io, value); }

void RAI_RDBSaveModel(RedisModuleIO *io, void *value) { RAI_RDBSaveModel_v2(io, value); }

void RAI_RDBSaveScript(RedisModuleIO *io, void *value) { RAI_RDBSaveScript_v2(io, value); }
