[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg?sort=semver)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)
[![Dockerhub](https://img.shields.io/badge/dockerhub-redislabs%2Fredisai-blue)](https://hub.docker.com/r/redislabs/redisai/tags/) 
[![codecov](https://codecov.io/gh/RedisAI/RedisAI/branch/master/graph/badge.svg)](https://codecov.io/gh/RedisAI/RedisAI)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/RedisAI/RedisAI.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RedisAI/RedisAI/alerts/)

# RedisAI
[![Forum](https://img.shields.io/badge/Forum-RedisAI-blue)](https://forum.redislabs.com/c/modules/redisai)
[![Discord](https://img.shields.io/discord/697882427875393627?style=flat-square)](https://discord.gg/rTQm7UZ)

A Redis module for serving tensors and executing deep learning models.

## Cloning
If you want to run examples, make sure you have [git-lfs](https://git-lfs.github.com) installed when you clone.

## Quickstart

1. [Docker](#docker)
2. [Build](#building)

## Docker

To quickly tryout RedisAI, launch an instance using docker:

```sh
docker run -p 6379:6379 -it --rm redislabs/redisai
```

For docker instance with GPU support, you can launch it from `tensorwerk/redisai-gpu`

```sh
docker run -p 6379:6379 --gpus all -it --rm redislabs/redisai:latest-gpu
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
redis-cli -x AI.MODELSET foo TF CPU INPUTS a b OUTPUTS c BLOB < tests/test_data/graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures:
* `AI.TENSORSET tensor_key data_type dim1..dimN [BLOB data | VALUES val1..valN]`
* `AI.MODELRUN graph_key INPUTS input_key1 ... OUTPUTS output_key1 ...`
```sh
redis-cli
> AI.TENSORSET bar FLOAT 2 VALUES 2 3
> AI.TENSORSET baz FLOAT 2 VALUES 2 3
> AI.MODELRUN foo INPUTS bar baz OUTPUTS jez
> AI.TENSORGET jez META VALUES
1) dtype
2) FLOAT
3) shape
4) 1) (integer) 2
5) values
6) 1) "4"
   2) "9"
```

## Building

You should obtain the module's source code and submodule using git like so: 

```sh
git clone --recursive https://github.com/RedisAI/RedisAI
```

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
ALL=1 make -C opt clean build
```

Note: in order to use the PyTorch backend on Linux, at least `gcc 4.9.2` is required.

### Running the server

You will need a redis-server version 5.0.7 or greater. This should be
available in most recent distributions:

```sh
redis-server --version
Redis server v=5.0.7 sha=00000000:0 malloc=libc bits=64 build=c49f4faf7c3c647a
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
| redisai-js | Typescript/Javascript | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/redisai-js) |

## Backend Dependancy

RedisAI currently supports PyTorch (libtorch), Tensorflow (libtensorflow), TensorFlow Lite, and ONNXRuntime as backends. This section shows the version map between RedisAI and supported backends. This extremely important since the serialization mechanism of one version might not match with another. For making sure your model will work with a given RedisAI version, check with the backend documentation about incompatible features between the version of your backend and the version RedisAI is built with.


| RedisAI | PyTorch | TensorFlow | TFLite | ONNXRuntime   |
|:--------|:-------:|:----------:|:------:|:-------------:|
| 0.1.0   | 1.0.1   | 1.12.0     | None   | None          |
| 0.2.1   | 1.0.1   | 1.12.0     | None   | None          |
| 0.3.1   | 1.1.0   | 1.12.0     | None   | 0.4.0         |
| 0.4.0   | 1.2.0   | 1.14.0     | None   | 0.5.0         |
| 0.9.0   | 1.3.1   | 1.14.0     | 2.0.0  | 1.0.0         |
| 1.0.0   | 1.5.0   | 1.15.0     | 2.0.0  | 1.2.0         |
| master  | 1.7.0   | 1.15.0     | 2.0.0  | 1.2.0         |

Note: Keras and TensorFlow 2.x are supported through graph freezing. See [this script](https://github.com/RedisAI/RedisAI/blob/master/tests/test_data/tf2-minimal.py) to see how to export a frozen graph from Keras and TensorFlow 2.x. Note that a frozen graph will be executed using the TensorFlow 1.15 backend. Should any 2.0 ops be not supported on the 1.15 after freezing, please open an Issue.

## Documentation

Read the docs at [redisai.io](http://redisai.io). Checkout our [showcase repo](https://github.com/RedisAI/redisai-examples) for a lot of examples written using different client libraries.

## Mailing List / Forum

Got questions? Feel free to ask at the [RedisAI Forum](https://forum.redislabs.com/c/modules/redisai)

## License

Redis Source Available License Agreement - see [LICENSE](LICENSE)

Copyright 2020, [Redis Labs, Inc](https://redislabs.com)
