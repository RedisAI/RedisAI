# RedisAI Module

RedisAI is a Redis module for serving tensors and executing deep learning models.

## Quickstart

1. [Docker](#docker)
2. [Build](#building)
3. [Start](#start)

## Docker

To quickly tryout RedisAI, launch an instance using docker:

```sh
docker run -p 6379:6379 -it --rm redisai/redisai
```

## Building
This will checkout and build Redis and download the libraries for the backends (TensorFlow and PyTorch) for your platform.
```
mkdir deps
DEPS_DIRECTORY=deps bash get_deps.sh

cd deps
git clone git://github.com/antirez/redis.git --branch 5.0
cd redis
make malloc=libc -j4
cd ../..

mkdir build
cd build
cmake -DDEPS_PATH=../deps/install ..
make
cd ..
```

## Start
If you want to run examples, make sure you have [git-lfs](https://git-lfs.github.com) installed when you clone.

On Linux
```
LD_LIBRARY_PATH=deps/install/lib deps/redis/src/redis-server --loadmodule build/redisai.so
```

On macos
```
deps/redis/src/redis-server --loadmodule build/redisai.so
```

On the client, load the model
```
./deps/redis/src/redis-cli -x AI.MODELSET foo TF CPU INPUTS a b OUTPUTS c < graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures: 
* `AI.TENSORSET tensor_key data_type dim1..dimN [BLOB data | VALUES val1..valN]`
* `AI.MODELRUN graph_key INPUTS input_key1 ... OUTPUTS output_key1 ...`
```
redis-cli
> AI.TENSORSET bar FLOAT 2 VALUES 2 3
> AI.TENSORSET baz FLOAT 2 VALUES 2 3
> AI.MODELRUN foo INPUTS bar baz OUTPUTS jez
> AI.TENSORGET jez VALUES
1) FLOAT
2) 1) (integer) 2
3) 1) "4"
   2) "9"
```

Full documentation of the api can be found [here](commands.md).
