#include "rai_rdb_encode.h"
#include "v100/encode_v100.h"

void RAI_RDBSaveTensor(RedisModuleIO *io, void *value) {
    RAI_RDBSaveTensor_v100(io, value);
}
