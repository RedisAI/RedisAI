#include <limits.h>
#include "gtest/gtest.h"

#include "rmalloc.h"
#include "src/dag.h"
#include <pthread.h>

// Declaration of thread condition variable
pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;

// declaring mutex
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void OnFinishAsync(RedisAI_OnFinishCtx on_finish_ctx, void *pdata) {

	// do stuff
	// release lock
	pthread_mutex_unlock(&lock);
}
class DagAsyncTest : public ::testing::Test {
protected:
	static void SetUpTestCase() {
		// Use the malloc family for allocations
		Alloc_Reset();
	}
};

TEST_F(DagAsyncTest, DAG_AsyncRunEnabled) {

	// Create a simple DAG Run info with a single TENSORSET operation
	// Equivalent to "AI.DAGRUN |> AI.TENSORSET mytensor FLOAT 1 2 VALUES 5 10"

	RedisAI_RunInfo *run_info;
	EXPECT_EQ(RAI_InitRunInfo(&run_info), REDISMODULE_OK);

}
