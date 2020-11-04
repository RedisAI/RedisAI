#include "rai_rdb_decoder.h"
#include "current/v100/decode_v100.h"

void* RAI_RDBLoadTensor( RedisModuleIO *io, int encver) {
    return RAI_RDBLoadTensor_v100(io);
}