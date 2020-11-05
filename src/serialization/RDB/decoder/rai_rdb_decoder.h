#pragma once

#include "../../serialization_include.h"

void* RAI_RDBLoadTensor(RedisModuleIO *io);

void* RAI_RDBLoadModel(RedisModuleIO *io);

void* RAI_RDBLoadScript(RedisModuleIO *io);