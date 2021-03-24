#include <limits.h>
#include "gtest/gtest.h"

extern "C" {
#include "rmalloc.h"
#include "src/redis_ai_objects/err.h"
}

class ErrorStructTest : public ::testing::Test {
    protected:
        static void SetUpTestCase() {
            // Use the malloc family for allocations
            Alloc_Reset();
        }
};

TEST_F(ErrorStructTest, RAI_GetError) {
    RAI_Error* err;
    EXPECT_EQ(0,RAI_InitError(&err));
    EXPECT_STREQ(RAI_GetError(err),nullptr);
    EXPECT_STREQ(RAI_GetErrorOneLine(err),nullptr);
    EXPECT_EQ(RAI_GetErrorCode(err),RAI_OK);
    RAI_FreeError(err);
}

TEST_F(ErrorStructTest, RAI_SetError_nullptr) {
    RAI_Error* err = nullptr;

    // Test for nullptr err
    RAI_SetError(err,RAI_OK,nullptr);

    // ensure RAI_FreeError on nullptr is safe
    RAI_FreeError(err);

    // ensure RAI_ClearError on nullptr is safe
    RAI_ClearError(err);
}

TEST_F(ErrorStructTest, RAI_SetError_default) {
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
