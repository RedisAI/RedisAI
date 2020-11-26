#include <benchmark/benchmark.h>
extern "C" {
#include "rmalloc.h"
#include "src/err.h"
#include "src/tensor.h"
}


static void BM_RAI_TensorCreate(benchmark::State& state) {
    Alloc_Reset();
    long long *dims=(long long *)malloc(4*sizeof( long long ));
    dims[0]=1;
    dims[1]=3;
    dims[2]=224;
    dims[3]=224;       
    while (state.KeepRunning())
    {
        // RAI_Tensor* t = RAI_TensorCreate("FLOAT",dims,4,TENSORALLOC_ALLOC);
          // read/write barrier
        benchmark::ClobberMemory();
    }
}

static void BM_RAI_TensorLength(benchmark::State& state) {
      Alloc_Reset();
      long long *dims=(long long *)malloc(4*sizeof( long long ));
    dims[0]=1;
    dims[1]=3;
    dims[2]=224;
    dims[3]=224;
    RAI_Tensor* t = RAI_TensorCreate("FLOAT",dims,4,TENSORALLOC_ALLOC);
    const size_t len = RAI_TensorLength(t);
    // double check the calculation is correct
    assert(len==150528);

    // microbenchmark timing starts here
    while (state.KeepRunning())
    {
        benchmark::DoNotOptimize(
           RAI_TensorLength(t)
        );
        // read/write barrier
        benchmark::ClobberMemory();
    }
}

// Register the functions as a benchmark
BENCHMARK(BM_RAI_TensorCreate);
BENCHMARK(BM_RAI_TensorLength);

BENCHMARK_MAIN();
