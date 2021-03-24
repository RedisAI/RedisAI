# RedisAI Quickstart
RedisAI is a Redis module. To run it you'll need a Redis server (v5.0.7 or greater), the module's shared library, and its dependencies.

The following sections describe how to get started with RedisAI.

## Docker
The quickest way to try RedisAI is by launching its official Docker container images:

### On a CPU only machine
```
docker run -p 6379:6379 redislabs/redisai:latest
```

### On a GPU machine

```
docker run -p 6379:6379 --gpus all -it --rm redislabs/redisai:latest-gpu
```

## Download

A pre-compiled version can be downloaded from [RedisLabs download center](https://redislabs.com/download-center/modules/).

## Building
You can compile and build the module from its source code. The [Developer](developer.md) page has more information about the design and implementation of the RedisAI module and how to contribute.

### Prerequisites
* Packages: git, python3, make, wget, g++/clang, & unzip 
* CMake 3.0 or higher needs to be installed.
* CUDA needs to be installed for GPU support.
* Redis v5.0.7 or greater.

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
Use the following script to download and build the libraries of the various RedisAI backends (TensorFlow, PyTorch, ONNXRuntime) for your platform with GPU support:

```sh
bash get_deps.sh
```

Alternatively, you can run the following to fetch the CPU-only backends.

```sh
bash get_deps.sh cpu
```

### Building the Module
Once the dependencies have been built, you can build the RedisAI module with:

```sh
make -C opt build
```

## Loading the Module
To load the module on the same server is was compiled on simply use the `--loadmodule` command line switch, the `loadmodule` configuration directive or the [Redis `MODULE LOAD` command](https://redis.io/commands/module-load) with the path to module's library.

For example, to load the module from the project's path with a server command line switch use the following:

```sh
redis-server --loadmodule ./install-cpu/redisai.so
```
