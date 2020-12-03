#include "rai_rdb_encode.h"
#include "v1/encode_v1.h"

void RAI_RDBSaveTensor(RedisModuleIO *io, void *value) { RAI_RDBSaveTensor_v1(io, value); }

void RAI_RDBSaveModel(RedisModuleIO *io, void *value) { RAI_RDBSaveModel_v1(io, value); }

void RAI_RDBSaveScript(RedisModuleIO *io, void *value) { RAI_RDBSaveScript_v1(io, value); }
