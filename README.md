# redis-tf

A Redis module for serving Tensorflow tensors and executing graphs.

## Building
This will checkout and build Redis and Tensorflow (you'll need bazel for the latter).
```
sh build_deps.sh
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
* `TF.TENSOR tensor_key data_type ndims dim1 dim2 ... [data]`
* `TF.RUN graph_key ninputs input_key input_name_in_graph ... output_key output_name_in_graph ...`
```
redis-cli
> TF.TENSOR bar UINT8 1 1 1
> TF.TENSOR baz UINT8 1 1 1
> TF.RUN foo 2 bar a baz b jez c
> TF.DATA jez
```

## Acknowledgements

[Jim Fleming's post](https://medium.com/jim-fleming/loading-tensorflow-graphs-via-host-languages-be10fd81876f#.lqyteltuo)

## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2016, Luca Antiga, Orobix Srl (www.orobix.com).

