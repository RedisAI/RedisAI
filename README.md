[![license](https://img.shields.io/github/license/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI)
[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)

# RedisAI

A Redis module for serving tensors and executing deep learning models.
Expect changes in the API and internals.

## Cloning
If you want to run examples, make sure you have [git-lfs](https://git-lfs.github.com) installed when you clone.

## Building
This will checkout and build Redis and download the libraries for the backends (TensorFlow and PyTorch) for your platform.
```
bash get_deps.sh
make install
```

## Docker

To quickly tryout RedisAI, launch an instance using docker:

```sh
docker run -p 6379:6379 -it --rm redisai/redisai
```

## Running the server
On Linux
```
LD_LIBRARY_PATH=deps/install/lib ./deps/redis/src/redis-server -loadmodule src/redisai.so
```

On macos
```
DYLD_LIBRARY_PATH=deps/install/lib ./deps/redis/src/redis-server -loadmodule src/redisai.so
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
3) 1) "2"
   2) "3"
```

## Documentation

Read the docs at [redisai.io](http://redisai.io).

## Mailing List

[RedisAI Google group](https://groups.google.com/forum/#!forum/redisai)

## License

AGPL-3.0 https://opensource.org/licenses/AGPL-3.0

Copyright 2019, Orobix Srl & Redis Labs
