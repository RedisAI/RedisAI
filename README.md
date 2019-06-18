[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/redisai/redisai.svg)](https://hub.docker.com/r/redisai/redisai/builds/)

# RedisAI

A Redis module for serving tensors and executing deep learning models.
Expect changes in the API and internals.

## Cloning
If you want to run examples, make sure you have [git-lfs](https://git-lfs.github.com) installed when you clone.

## Quickstart

1. [Docker](#docker)
2. [Build](#building)

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
(TensorFlow, PyTorch, ONNXRuntime) for your platform.

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

### Running the server

You will need a redis-server version 4.0.9 or greater. This should be
available in most recent distributions:

```sh
redis-server --version
Redis server v=4.0.9 sha=00000000:0 malloc=libc bits=64 build=c49f4faf7c3c647a
```

To start redis with the RedisAI module loaded:

```sh
redis-server --loadmodule build/redisai.so
```

## Client libraries

Some languages have client libraries that provide support for RedisAI's commands:

| Project | Language | License | Author | URL |
| ------- | -------- | ------- | ------ | --- |
| JRedisAI | Java | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/JRedisAI) |
| redisai-py | Python | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/redisai-py) |

## Backend Dependancy

RedisAI currently supports PyTorch (libtorch), Tensorflow (libtensorflow) and ONNXRuntime as backends. This section shows the version map between RedisAI and supported backends. This extremely important since the serialization mechanism of one version might not match with another. For making sure your model will work with a given RedisAI version, check with the backend documentation about incompatible features between the version of your backend and the version RedisAI is built with.


| RedisAI | PyTorch | TensorFlow | ONNXRuntime   |
|:--------|:-------:|:----------:|:-------------:|
| 0.1.0   | 1.0.1   | 1.12.0     | 0.4.0         |


## Documentation

Read the docs at [redisai.io](http://redisai.io).

## Mailing List

[RedisAI Google group](https://groups.google.com/forum/#!forum/redisai)

## License

Redis Source Available License Agreement - see [LICENSE](LICENSE)

Copyright 2019, Tensorwerk Inc & Redis Labs Ltd
