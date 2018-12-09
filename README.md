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
./deps/redis/src/redis-cli -x DL.SET GRAPH foo TF < graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures: 
* `DL.SET TENSOR tensor_key data_type ndims dim1..dimN [BLOB data | VALUES val1..valN]`
* `DL.RUN GRAPH graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...`
```
redis-cli
> DL.SET TENSOR bar FLOAT 1 2 VALUES 2 3
> DL.SET TENSOR baz FLOAT 1 2 VALUES 2 3
> DL.RUN GRAPH foo 2 bar a baz b jez c
> DL.GET TENSOR jez VALUES
1) FLOAT
2) (integer) 1
3) 1) (integer) 2
4) (integer) 8
5) 1) "2"
   2) "3"
```

### DL.SET TENSOR tensor_key data_type dim shape1..shapeN [BLOB data | VALUES val1..valN]
Stores a tensor of defined type (FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16) with N dimensions (dim) and shape given by shape1..shapeN

### DL.SET GRAPH graph_key backend graph_blob prefix
Stores a graph provided as a protobuf blob. Backend is TF for now.

### DL.GET TENSOR tensor_key [BLOB | VALUES | META]

### DL.RUN GRAPH graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...


## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2018, Luca Antiga, Orobix Srl (www.orobix.com).
