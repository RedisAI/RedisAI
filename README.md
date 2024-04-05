[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg?sort=semver)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)
[![Dockerhub](https://img.shields.io/badge/dockerhub-redislabs%2Fredisai-blue)](https://hub.docker.com/r/redislabs/redisai/tags/)
[![codecov](https://codecov.io/gh/RedisAI/RedisAI/branch/master/graph/badge.svg)](https://codecov.io/gh/RedisAI/RedisAI)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/RedisAI/RedisAI.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RedisAI/RedisAI/alerts/)
[![Forum](https://img.shields.io/badge/Forum-RedisAI-blue)](https://forum.redislabs.com/c/modules/redisai)
[![Discord](https://img.shields.io/discord/697882427875393627?style=flat-square)](https://discord.gg/rTQm7UZ)

> [!CAUTION]
> **RedisAI is no longer actively maintained or supported.**
>
> We are grateful to the RedisAI community for their interest and support.

# RedisAI
RedisAI is a Redis module for executing Deep Learning/Machine Learning models and managing their data. Its purpose is being a "workhorse" for model serving, by providing out-of-the-box support for popular DL/ML frameworks and unparalleled performance. **RedisAI both maximizes computation throughput and reduces latency by adhering to the principle of data locality**, as well as simplifies the deployment and serving of graphs by leveraging on Redis' production-proven infrastructure.

To read RedisAI docs, visit [redisai.io](https://oss.redis.com/redisai/). To see RedisAI in action, visit the [demos page](https://oss.redis.com/redisai/examples/). 

# Quickstart
RedisAI is a Redis module. To run it you'll need a Redis server (v6.0.0 or greater), the module's shared library, and its dependencies.

The following sections describe how to get started with RedisAI.

## Docker
The quickest way to try RedisAI is by launching its official Docker container images.
### On a CPU only machine
```
docker run -p 6379:6379 redislabs/redisai:1.2.7-cpu-bionic
```

### On a GPU machine
For GPU support you will need a machine you'll need a machine that has Nvidia driver (CUDA 11.3 and cuDNN 8.1), nvidia-container-toolkit and Docker 19.03+ installed. For detailed information, checkout [nvidia-docker documentation](https://github.com/NVIDIA/nvidia-docker)

```
docker run -p 6379:6379 --gpus all -it --rm redislabs/redisai:1.2.7-gpu-bionic
```


## Building
You can compile and build the module from its source code. The [Developer](https://oss.redis.com/redisai/developer/) page has more information about the design and implementation of the RedisAI module and how to contribute.

### Prerequisites
* Packages: git, python3, make, wget, g++/clang, & unzip
* CMake 3.0 or higher needs to be installed.
* CUDA 11.3 and cuDNN 8.1 or higher needs to be installed if GPU support is required.
* Redis v6.0.0 or greater.

### Get the Source Code
You can obtain the module's source code by cloning the project's repository using git like so:

```sh
git clone --recursive https://github.com/RedisAI/RedisAI
```

Switch to the project's directory with:

```sh
cd RedisAI
```

### Building the Dependencies
Use the following script to download and build the libraries of the various RedisAI backends (TensorFlow, PyTorch, ONNXRuntime) for CPU only:

```sh
bash get_deps.sh
```

Alternatively, you can run the following to fetch the backends with GPU support.

```sh
bash get_deps.sh gpu
```

### Building the Module
Once the dependencies have been built, you can build the RedisAI module with:

```sh
make -C opt clean ALL=1
make -C opt
```

Alternatively, run the following to build RedisAI with GPU support:

```sh
make -C opt clean ALL=1
make -C opt GPU=1
```

### Backend Dependancy

RedisAI currently supports PyTorch (libtorch), Tensorflow (libtensorflow), TensorFlow Lite, and ONNXRuntime as backends. This section shows the version map between RedisAI and supported backends. This extremely important since the serialization mechanism of one version might not match with another. For making sure your model will work with a given RedisAI version, check with the backend documentation about incompatible features between the version of your backend and the version RedisAI is built with.


| RedisAI | PyTorch  | TensorFlow | TFLite | ONNXRuntime |
|:--------|:--------:|:----------:|:------:|:-----------:|
| 1.0.3   |  1.5.0   |   1.15.0   | 2.0.0  |    1.2.0    |
| 1.2.7   |  1.11.0  |   2.8.0    | 2.0.0  |   1.11.1    |
| master  |  1.11.0  |   2.8.0    | 2.0.0  |   1.11.1    |

Note: Keras and TensorFlow 2.x are supported through graph freezing. See [this script](http://dev.cto.redis.s3.amazonaws.com/RedisAI/test_data/tf2-minimal.py
) to see how to export a frozen graph from Keras and TensorFlow 2.x.

## Loading the Module
To load the module upon starting the Redis server, simply use the `--loadmodule` command line switch, the `loadmodule` configuration directive or the [Redis `MODULE LOAD` command](https://redis.io/commands/module-load) with the path to module's library.

For example, to load the module from the project's path with a server command line switch use the following:

```sh
redis-server --loadmodule ./install-cpu/redisai.so
```

### Give it a try

Once loaded, you can interact with RedisAI using redis-cli. Basic information and examples for using the module is described [here](https://oss.redis.com/redisai/intro/#getting-started).

### Client libraries
Some languages already have client libraries that provide support for RedisAI's commands. The following table lists the known ones:

| Project            | Language              | License      | Author                                           | URL                                                         |
| -------            | --------              | -------      | ------                                           | ---                                                         |
| JRedisAI           | Java                  | BSD-3        | [RedisLabs](https://redislabs.com/)              | [Github](https://github.com/RedisAI/JRedisAI)               |
| redisai-py         | Python                | BSD-3        | [RedisLabs](https://redislabs.com/)              | [Github](https://github.com/RedisAI/redisai-py)             |
| redisai-go         | Go                    | BSD-3        | [RedisLabs](https://redislabs.com/)              | [Github](https://github.com/RedisAI/redisai-go)             |
| redisai-js         | Typescript/Javascript | BSD-3        | [RedisLabs](https://redislabs.com/)              | [Github](https://github.com/RedisAI/redisai-js)             |
| redis-modules-sdk  | TypeScript            | BSD-3-Clause | [Dani Tseitlin](https://github.com/danitseitlin) | [Github](https://github.com/danitseitlin/redis-modules-sdk) |
| redis-modules-java | Java                  | Apache-2.0   | [dengliming](https://github.com/dengliming)      | [Github](https://github.com/dengliming/redis-modules-java)  |
| smartredis         | C++                   | BSD-2-Clause | [Cray Labs](https://github.com/CrayLabs)         | [Github](https://github.com/CrayLabs/SmartRedis)            |
| smartredis         | C                     | BSD-2-Clause | [Cray Labs](https://github.com/CrayLabs)         | [Github](https://github.com/CrayLabs/SmartRedis)            |
| smartredis         | Fortran               | BSD-2-Clause | [Cray Labs](https://github.com/CrayLabs)         | [Github](https://github.com/CrayLabs/SmartRedis)            |
| smartredis         | Python                | BSD-2-Clause | [Cray Labs](https://github.com/CrayLabs)         | [Github](https://github.com/CrayLabs/SmartRedis)            |



The full documentation for RedisAI's API can be found at the [Commands page](commands.md).

## Documentation
Read the docs at [redisai.io](https://oss.redis.com/redisai/).

## Contact Us
If you have questions, want to provide feedback or perhaps report an issue or [contribute some code](contrib.md), here's where we're listening to you:

* [Forum](https://forum.redis.com/c/modules/redisai)
* [Repository](https://github.com/RedisAI/RedisAI/issues)

## License
RedisAI is licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or the Server Side Public License v1 (SSPLv1).
