<img src="https://camo.githubusercontent.com/8188ec467c9580f0c9614df2bfdc26e1b471d0c5/68747470733a2f2f63646e342e69636f6e66696e6465722e636f6d2f646174612f69636f6e732f72656469732d322f313435312f556e7469746c65642d322d3531322e706e67" width="256">
<img src="http://www.clipartbest.com/cliparts/Bdc/reK/BdcreKoT9.jpeg" width="256">
<img src="https://avatars0.githubusercontent.com/u/15658638?v=3&s=400" width="256">

# RedisTF

A Redis module for serving Tensorflow tensors and executing graphs.

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
cd models
# build a graph and write it out
python tf-minimal.py
```

On the client, load the graph
```
redis-cli -x TF.GRAPH foo < graph.pb
```

Then create the input tensors, run the computation graph and get the output tensor (see `load_model.sh`). Note the signatures: 
* `TF.TENSOR tensor_key data_type ndims dim1..dimN [BLOB data | VALUES val1..valN]`
* `TF.RUN graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...`
```
redis-cli
> TF.TENSOR bar FLOAT 1 2 VALUES 2 3
> TF.TENSOR baz FLOAT 1 2 VALUES 2 3
> TF.RUN foo 2 bar a baz b jez c
> TF.VALUES jez
1) (integer) 4
2) (integer) 9
```

## Acknowledgements

[Jim Fleming's post](https://medium.com/jim-fleming/loading-tensorflow-graphs-via-host-languages-be10fd81876f#.lqyteltuo)

## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2018, Luca Antiga, Orobix Srl (www.orobix.com).

