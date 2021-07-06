#include "rai_rdb_encode.h"
#include "v3/encode_v3.h"

void RAI_RDBSaveTensor(RedisModuleIO *io, void *value) { RAI_RDBSaveTensor_v3(io, value); }

void RAI_RDBSaveModel(RedisModuleIO *io, void *value) { RAI_RDBSaveModel_v3(io, value); }

void RAI_RDBSaveScript(RedisModuleIO *io, void *value) { RAI_RDBSaveScript_v3(io, value); }
