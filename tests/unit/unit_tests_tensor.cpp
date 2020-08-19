#include <limits.h>
#include "gtest/gtest.h"

extern "C" {
#include "src/redismodule.h"
#include "src/err.h"
}

namespace {

void Alloc_Reset() {
    RedisModule_Alloc = malloc;
    RedisModule_Realloc = realloc;
    RedisModule_Calloc = calloc;
    RedisModule_Free = free;
    RedisModule_Strdup = strdup;
}

TEST(tensor, RAI_GetError) {
    Alloc_Reset();
    RAI_Error* err;
    EXPECT_EQ(0,RAI_InitError(&err));
    EXPECT_STREQ(RAI_GetError(err),nullptr);
    EXPECT_STREQ(RAI_GetErrorOneLine(err),nullptr);
    EXPECT_EQ(RAI_GetErrorCode(err),RAI_OK);
    RAI_FreeError(err);
}

TEST(err, RAI_SetError_nullptr) {
    Alloc_Reset();
    RAI_Error* err = nullptr;

    // Test for nullptr err
    RAI_SetError(err,RAI_OK,nullptr);

    // ensure RAI_FreeError on nullptr is safe
    RAI_FreeError(err);

    // ensure RAI_ClearError on nullptr is safe
    RAI_ClearError(err);
}

TEST(err, RAI_SetError_default) {
    Alloc_Reset();
    RAI_Error* err;
    EXPECT_EQ(0,RAI_InitError(&err));
    RAI_SetError(err,RAI_OK,nullptr);
    EXPECT_STREQ(RAI_GetErrorOneLine(err),"ERR Generic error");
    EXPECT_EQ(RAI_GetErrorCode(err),RAI_OK);

    // clear error to reuse structure
    RAI_ClearError(err);

    RAI_SetError(err,RAI_EMODELCREATE,"ERR specific error message");
    EXPECT_STREQ(RAI_GetErrorOneLine(err),"ERR specific error message");
    EXPECT_EQ(RAI_GetErrorCode(err),RAI_EMODELCREATE);

    RAI_FreeError(err);
}

}  // namespace
