#include "err.h"
#include "stdlib.h"
#include "assert.h"
#include "string.h"

#include "redismodule.h"

void RAI_SetError(RAI_Error *err, RAI_ErrorCode code, const char *detail) {
  if (err->code != RAI_OK) {
    return;
  }
  assert(!err->detail);
  err->code = code;

  if (detail) {
    err->detail = RedisModule_Strdup(detail);
  } else {
    err->detail = strdup("Generic error");
  }
}

RAI_Error RAI_InitError() {
  RAI_Error err = {.code = RAI_OK, .detail = NULL};
  return err;
}

void RAI_ClearError(RAI_Error *err) {
  if (err->detail) {
    free(err->detail);
    err->detail = NULL;
  }
  err->code = RAI_OK;
}
