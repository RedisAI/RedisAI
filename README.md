[![license](https://img.shields.io/github/license/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI)
[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)

# RedisAI

A Redis module for serving tensors and executing deep learning graphs.
Expect changes in the API and internals.

## Cloning
If you want to run examples, make sure you have [git-lfs](https://git-lfs.github.com) installed when you clone.

## Building
This will checkout and build Redis and download the libtensorflow binaries for your platform.
```
bash get_deps.sh
make install
```

## Running the server
```
make run
```

## Running tests
Tests are in the works, there's just a check script (requires Python 3.6.x)
Make sure the server is running, then
```
cd examples/models
# build a graph and write it out
python tf-minimal.py
```

On the client, load the graph
```
./deps/redis/src/redis-cli -x AI.SET GRAPH foo TF < graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures: 
* `AI.SET TENSOR tensor_key data_type ndims dim1..dimN [BLOB data | VALUES val1..valN]`
* `AI.RUN GRAPH graph_key INPUTS ninputs input_key1 ... NAMES input_name_in_graph1 ... OUTPUTS noutputs output_key1 ... NAMES output_name_in_graph1 ...`
```
redis-cli
> AI.SET TENSOR bar FLOAT 1 2 VALUES 2 3
> AI.SET TENSOR baz FLOAT 1 2 VALUES 2 3
> AI.RUN GRAPH foo INPUTS 2 bar baz NAMES a b OUTPUTS 1 jez NAMES c
> AI.GET TENSOR jez VALUES
1) FLOAT
2) (integer) 1
3) 1) (integer) 2
4) (integer) 8
5) 1) "2"
   2) "3"
```

## Documentation

Read the docs at [redisai.io](http://redisai.io).

## Mailing List

[RedisAI Google group](https://groups.google.com/forum/#!forum/redisai)

## License

AGPL-3.0 https://opensource.org/licenses/AGPL-3.0

Copyright 2018, Orobix Srl & Redis Labs
