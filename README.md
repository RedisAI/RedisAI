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
./deps/redis/src/redis-cli -x DL.GSET foo TF < graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures: 
* `DL.TSET tensor_key data_type ndims dim1..dimN [BLOB data | VALUES val1..valN]`
* `DL.GRUN graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...`
```
redis-cli
> DL.TSET bar FLOAT 1 2 VALUES 2 3
> DL.TSET baz FLOAT 1 2 VALUES 2 3
> DL.GRUN foo 2 bar a baz b jez c
> DL.TGET jez VALUES
1) (integer) 4
2) (integer) 9
```

### DL.TSET tensor_key data_type dim shape1..shapeN [BLOB data | VALUES val1..valN]
Stores a tensor of defined type (FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16) with N dimensions (dim) and shape given by shape1..shapeN

### DL.TGET tensor_key [BLOB | VALUES]

### DL.TDATATYPE tensor_key

### DL.TDIM tensor_key

### DL.TSHAPE tensor_key

### DL.TBYTESIZE tensor_key

### DL.GSET graph_key TF graph_blob prefix
Stores a graph provided as a protobuf blob

### DL.GRUN graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...


## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2018, Luca Antiga, Orobix Srl (www.orobix.com).
