<img src="images/logo.svg" alt="logo" width="200"/>

# RedisAI

RedisAI is a Redis module for serving tensors and executing deep learning models.

## Quickstart

1. [Docker](#docker)
2. [Build](#building)

## Docker

To quickly tryout RedisAI, launch an instance using docker:

```sh
docker run -p 6379:6379 -it --rm redisai/redisai
```

### Give it a try

On the client, load a backend (TF, TORCH or ONNX), and set the model
```sh
redis-cli AI.CONFIG LOADBACKEND TF install/backends/redisai_tensorflow/redisai_tensorflow.so
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
This will checkout and build and download the libraries for the backends
(TensorFlow, PyTorch, ONNXRuntime) for your platform.

```sh
bash get_deps.sh

```

After the dependencies are downloaded, build the module itself. Note that
CMake 3.0 or higher is required.

```sh
mkdir build
cd build
cmake -DDEPS_PATH=../deps/install ..
make && make install
cd ..
```

### Running the server

You must have a redis-server version 4.0.9 or greater, available in most recent distributions:

```sh
redis-server --version
Redis server v=4.0.9 sha=00000000:0 malloc=libc bits=64 build=c49f4faf7c3c647a
```

To start Redis with the RedisAI module loaded:

```
redis-server --loadmodule install/redisai.so
```

## Client libraries

Some languages have client libraries that provide support for RedisAI's commands:

| Project | Language | License | Author | URL |
| ------- | -------- | ------- | ------ | --- |
| JRedisAI | Java | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/JRedisAI) |
| redisai-py | Python | BSD-3 | [RedisLabs](https://redislabs.com/) | [Github](https://github.com/RedisAI/redisai-py) |

Full documentation of the api can be found [here](commands.md).
