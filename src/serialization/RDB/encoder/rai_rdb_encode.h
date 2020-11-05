#pragma once

#include "../../serialization_include.h"

void RAI_RDBSaveTensor(RedisModuleIO *io, void *value);

void RAI_RDBSaveModel(RedisModuleIO *io, void *value);

void RAI_RDBSaveScript(RedisModuleIO *io, void *value);
