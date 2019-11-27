[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/redisai/redisai.svg)](https://hub.docker.com/r/redisai/redisai/builds/)
[![Mailing List](https://img.shields.io/badge/Mailing%20List-RedisAI-blue)](https://groups.google.com/forum/#!forum/redisai)
[![Gitter](https://badges.gitter.im/RedisLabs/RedisAI.svg)](https://gitter.im/RedisLabs/RedisAI?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


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

For docker instance with GPU support, you can launch it from `tensorwerk/redisai-gpu`

```sh
docker run -p 6379:6379 --gpus all -it --rm redisai/redisai:latest-gpu
```

But if you'd like to build the docker image, you need a machine that has Nvidia driver (CUDA 10.0), nvidia-container-toolkit and Docker 19.03+ installed. For detailed information, checkout [nvidia-docker documentation](https://github.com/NVIDIA/nvidia-docker)

```sh
docker build -f Dockerfile-gpu -t redisai-gpu .
docker run -p 6379:6379 --gpus all -it --rm redisai-gpu
```

Note that Redis config is located at `/usr/local/etc/redis/redis.conf` which can be overridden with a volume mount


### Give it a try

On the client, set the model
```sh
redis-cli -x AI.MODELSET foo TF CPU INPUTS a b OUTPUTS c < test/test_data/graph.pb
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
This will checkout and build and download the libraries for the backends (TensorFlow, PyTorch, ONNXRuntime) for your platform. Note that this requires CUDA 10.0 to be installed.

```sh
bash get_deps.sh
```

Alternatively, run the following to only fetch the CPU-only backends even on GPU machines.
```sh
bash get_deps.sh cpu
```

Once the dependencies are downloaded, build the module itself. Note that
CMake 3.0 or higher is required.

```sh
mkdir build
cd build
cmake ..
make && make install
cd ..
```

Note: in order to use the PyTorch backend on Linux, at least `gcc 4.9.2` is required.

### Running the server

You will need a redis-server version 4.0.9 or greater. This should be
available in most recent distributions:

```sh
redis-server --version
Redis server v=4.0.9 sha=00000000:0 malloc=libc bits=64 build=c49f4faf7c3c647a
```

To start Redis with the RedisAI module loaded:

```sh
redis-server --loadmodule install-cpu/redisai.so
```

## Client libraries

Some languages have client libraries that provide support for RedisAI's commands:

| Project | Language | License | Author | URL |
| ------- | -------- | ------- | ------ | --- |
| JRedisAI | Java | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/JRedisAI) |
| redisai-py | Python | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/redisai-py) |
| redisai-go | Go | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/redisai-go) |

## Backend Dependancy

RedisAI currently supports PyTorch (libtorch), Tensorflow (libtensorflow) and ONNXRuntime as backends. This section shows the version map between RedisAI and supported backends. This extremely important since the serialization mechanism of one version might not match with another. For making sure your model will work with a given RedisAI version, check with the backend documentation about incompatible features between the version of your backend and the version RedisAI is built with.


| RedisAI | PyTorch | TensorFlow | ONNXRuntime   |
|:--------|:-------:|:----------:|:-------------:|
| 0.1.0   | 1.0.1     | 1.12.0     | None          |
| 0.2.1   | 1.0.1     | 1.12.0     | None          |
| 0.3.1   | 1.1.0     | 1.12.0     | 0.4.0         |
| 0.4.0   | 1.2.0     | 1.14.0     | 0.5.0         |
| master  | 1.3.1     | 1.14.0     | 1.0.0         |


## Documentation

Read the docs at [redisai.io](http://redisai.io). Checkout our [showcase repo](https://github.com/RedisAI/redisai-examples) for a lot of examples written using different client libraries.

## Mailing List

[RedisAI Google group](https://groups.google.com/forum/#!forum/redisai)

## License

Redis Source Available License Agreement - see [LICENSE](LICENSE)

Copyright 2019, Tensorwerk Inc & Redis Labs Ltd
