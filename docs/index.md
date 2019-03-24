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

### Give it a try

On the client, load the model
```sh
redis-cli -x AI.MODELSET foo TF CPU INPUTS a b OUTPUTS c < examples/models/graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures:
* `AI.TENSORSET tensor_key data_type dim1..dimN [BLOB data | VALUES val1..valN]`
* `AI.MODELRUN graph_key INPUTS input_key1 ... OUTPUTS output_key1 ...`
```sh
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

## Building
This will checkout and build and download the libraries for the backends
(TensorFlow and PyTorch) for your platform.

```sh
bash get_deps.sh

```

Once the dependencies are downloaded, build the module itself. Note that
CMake 3.0 or higher is required.

```sh
mkdir build
cd build
cmake -DDEPS_PATH=../deps/install ..
make
cd ..
```

## Start
You will need a redis-server version 4.0.9 or greater. This should be
available in most recent distributions:

```sh
redis-server --version
Redis server v=4.0.9 sha=00000000:0 malloc=libc bits=64 build=c49f4faf7c3c647a
```

To start redis with the RedisAI module loaded, you need to make sure the dependencies can be found by redis.  One example on how to do this on Linux is:

```
LD_LIBRARY_PATH=<PATH_TO>/deps/install/lib redis-server --loadmodule build/redisai.so
```

Full documentation of the api can be found [here](commands.md).
