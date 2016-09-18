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
# load the graph, create tensors, run the graph and get the output tensor
sh load_model.sh
```

## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2016, Luca Antiga, Orobix Srl (www.orobix.com).

