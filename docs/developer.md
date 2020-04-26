# RedisAI Development

The following sections discusses topics relevant to the development of the RedisAI module itself. We'll start by refering to the general design, followed by the source code layout, and how to prepare your local test and development environment.

## General Design

RedisAI bundles together best-of-breed technologies for delivering stable and performant model serving. To do so, we need to abstract from what each specific DL/ML framework offers and provide common data structures and APIs to the DL/ML domain. 


As a way of representing tensor data we've embraced [dlpack](https://github.com/dmlc/dlpack) - a community effort to define a common tensor data structure that can be shared by different frameworks, supported by cuPy, cuDF, DGL, TGL, PyTorch, and MxNet.

### Data Structures
RedisAI provides the following data structures:

* **Tensor**: represents an n-dimensional array of values
* **Model**: represents a frozen graph by one of the supported DL/ML framework backends
* **Script**: represents a [TorchScript](https://pytorch.org/docs/stable/jit.html)

TBD

## Source code layout

Inside the root are the following important directories:

* `src`: contains the RedisAI implementation, written in C.
* `opt`: contains the helper scriptsto setup the development environment.
* `tests`: contains the unit and flow tests, implemented with python and RLTest.
* `deps`: contains libraries RedisAI uses.

We'll focus mostly on `src`, where the RedisAI implementation is contained,
exploring what there is inside each file. The order in which files are
exposed is the logical one to follow in order to disclose different layers
of complexity incrementally.

### redisai.h
---
TBD...

### redisai.c 
---

This is the entry point of the RedisAI module, TBD...

### tensor.c
---
TBD...

### model.c
---
TBD...

### script.c
---
TBD...

### dag.c
---
TBD...


### run_info.c
---
TBD...

### model_script_run_session.c
---
TBD...

### background_workers.c
---
TBD...

### Backends directory
---
TBD...

### Other C files
---

TBD...


## Building and Testing
You can compile and build the module from its source code.

### Prerequisites
* CUDA needs to be installed for GPU support.
* CMake 3.0 or higher needs to be installed.
* Redis v4.0.9 or greater.

### Get the Source Code
You can obtain the module's source code by cloning the project's repository using git like so:

```sh
git clone https://github.com/RedisAI/RedisAI
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

### Running Tests

The module includes a basic set of unit tests and integration tests, split across common and backend specific files. To run
them you'll need:

* lcov (for coverage analysis, on Linux)
* Python and a few packages (e.g. pytest, redis, etc.)
* redis-server in your PATH, or in `../redis/src`.

To run tests in a Python virtualenv, follow these steps:

    $ mkdir -p .env
    $ virtualenv .env
    $ . .env/bin/active
    $ pip install -r tests/tests_requirements.txt
    $ make -C opt tests

Integration tests are based on RLTest, and specific setup parameters can be provided
to configure tests.

For example, ... ( TBD )

### Unit Test Coverage

TBD

### Integration Tests Coverage

TBD 

