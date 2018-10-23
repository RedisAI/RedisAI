# RedisDL

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
Tests are in the works, there's just a check script.
Make sure the server is running, then
```
cd examples/models
# build a graph and write it out
python tf-minimal.py
```

On the client, load the graph
```
./deps/redis/src/redis-cli -x DL.GRAPH foo < graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures: 
* `DL.TENSOR tensor_key data_type ndims dim1..dimN [BLOB data | VALUES val1..valN]`
* `DL.RUN graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...`
```
redis-cli
> DL.TENSOR bar FLOAT 1 2 VALUES 2 3
> DL.TENSOR baz FLOAT 1 2 VALUES 2 3
> DL.RUN foo 2 bar a baz b jez c
> DL.VALUES jez
1) (integer) 4
2) (integer) 9
```

### DL.TENSOR tensor_key data_type ndims dim1..dimN [BLOB data | VALUES val1..valN]
Stores a tensor of defined type (FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16) with n dimensions (ndims)  

### DL.RUN graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...

### DL.TYPE tensor_key

### DL.NDIMS tensor_key

### DL.SIZE tensor_key

### DL.DATA tensor_key

### DL.VALUES tensor_key

### DL.graph graph_key graph_blob prefix


## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2018, Luca Antiga, Orobix Srl (www.orobix.com).

