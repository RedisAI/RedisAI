# RedisAI Development

The following sections discuss topics relevant to the development of the RedisAI module itself. We'll start by referring to the general design, followed by the source code layout, and how to prepare your local test and development environment.

## General Design

RedisAI bundles together best-of-breed technologies for delivering stable and fast model serving. To do so, we need to abstract from what each specific DL/ML framework offers and provide common data structures and APIs to the DL/ML domain.

As a way of representing tensor data we've embraced [dlpack](https://github.com/dmlc/dlpack) - a community effort to define a common tensor data structure that can be shared by different frameworks, supported by cuPy, cuDF, DGL, TGL, PyTorch, and MxNet.

**Data Structures**

RedisAI provides the following data structures:

* **Tensor**: represents an n-dimensional array of values
* **Model**: represents a computation graph by one of the supported DL/ML framework backends
* **Script**: represents a [TorchScript](https://pytorch.org/docs/stable/jit.html) program

## Source code layout

Inside the root are the following important directories:

* `src`: contains the RedisAI implementation, written in C.
* `opt`: contains the helper scripts to set up the development environment.
* `tests`: contains the unit and flow tests, implemented with python and RLTest.
* `deps`: contains libraries RedisAI uses.

We'll focus mostly on `src`, where the RedisAI implementation is contained,
exploring what there is inside each file. The order in which files are
exposed is the logical one to follow in order to disclose different layers
of complexity incrementally.

**redisai.c**

This is the entry point of the RedisAI module, responsible for registering the new commands and types in the Redis server, and containing all command functions to be called. This file is also responsible for exporting of RedisAI objects (Tensor, Script, Model and DAG) low-level-APIs to other Redis modules.

**redisai.h**

The header file that contains the module's low-level-API. This file should be copied to a Redis module that plan on using RedisAI objects and functionality via low-level-API. Note that every function in redisai.h is named "RedisAI_{X}", and detailed description for it can be found under the name "RAI_{X}" in RedisAI header files.  

**redis_ai_types directory**

Contains the callbacks that are required for the new Tensor, Model and Script types. These callbacks are used by Redis server for data management and persistence.   

**redis_ai_objects**

Contains the internal implementation of the basic RedisAI objects - Tensor, Model and Script.
For each object there is a header file that contains the helper methods for both creating, populating, managing and freeing the data structure.
This also includes `stats.h` where you can find the structure and headers that create, initialize, get, reset, and free run-time statics, like call count, error count, and aggregate durations of Model and Script execution sessions. Note that the statistics are ephemeral, meaning that they are not persisted to the Redis key space, and are reset when the server is restarted.

**execution**

Contains the files and logic that are responsible for RedisAI execution commands. The structure of RedisAI "execution plan" can be found in `run_info.h` along with the headers that create, initialize, get, and free this structure .
Every execution is represented as a DAG (directional acyclic graph) of one or more operation. Execution requests are queued and executed asynchronously. `run_queue_info.h` contains the structure for managing per-device queues that are used for decoupling the work from the main thread to the background worker threads.
`background_workers.c` contains the loop that every worker runs in the process of execution requests.    
The execution directory contains the following sub-directories:

* DAG - contains the methods for creating, initializing and freeing DAG operations, methods for building DAG from low-level API, and methods for running the DAG commands in the background, and replying to DAG structured commands. Also, this contains methods for validating the entire DAG and sending its operation to the appropriate execution queues.
* execution_context - contains the structure and methods for running an instance of a Model and Script in RedisAI.
* parsing - here we can find the parsing logic execution related RedisAI commands.

**config**

Contains methods for parsing, retrieving and setting RedisAI configuration parameters (both load time and run time) and their initial values. 

**serialization**

Contains methods for serializing RedisAI types by Redis in RDB load, RDB save and AOF rewrite routines. 

**backends**

Contains the interface for supporting a backend library that can be loaded by the module.

The usage in a certain backend capabilities is enabled when it is implement and export (a subset of) the following methods that appear in `backends.h`:

* `model_create_with_nodes`: A callback function pointer that creates a
  model given the `RAI_ModelOpts` and input and output nodes.
* `model_create`: A callback function pointer that creates a model given
the `RAI_ModelOpts`.
* `model_run`: A callback function pointer that runs a model given the
`RAI_ModelRunCtx` pointer.
* `model_free`:  A callback function pointer that frees a model given the
  `RAI_Model` pointer
* `model_serialize`: A callback function pointer that serializes a model
given the `RAI_Model` pointer.
* `script_create`: A callback function pointer that creates a script.
* `script_free`: A callback function pointer that frees a script given
the `RAI_Script` pointer.
* `script_run`: A callback function pointer that runs a script given the
`RAI_ScriptRunCtx` pointer.

Note: there are additional methods to retrieve backend info and apply advanced features that appear in `backends.h`. These methods can be implemented in the backend and exported as well.

This directory also include the implementations code required to support the following DL/ML identifiers and respective backend libraries:

* **TF**: `tensorflow.h` and `tensorflow.c` exporting the functions to register the TensorFlow backend
* **TFLITE**: `tflite.h` and `tflite.c` exporting the functions to register the TensorFlow Lite backend
* **TORCH**: `torch.h` and `torch.c` exporting the functions to register the PyTorch backend. This is the only backend that can create and run scripts.
* **ONNX**: `onnxruntime.h` and `onnxruntime.c` exporting the functions to register the ONNXRuntime backend. This backend has several additional capabilities. First, it uses Redis allocator for its memory management. Therefore, the backend memory consumption can be monitored with `INFO MODULES` command, and limited with `BACKEND_MEMORY_LIMIT` configuration. Moreover, Onnxruntime has a "kill switch" mechanism that allows setting a global timeout for running models with `MODEL_EXECUTION_TIMEOUT` configuration, after which execution is terminated immediately. 

## Building and Testing

You can compile and build the module from its source code - refer to the [Building and Running section](quickstart.md#building-and-running) of the Quickstart page for instructions on how to do that.

### Configuring your system

**Building in a docker (x86_64)**

The RedisAI source code can be mounted in a docker, and built there, but edited from the external operating system. This assumes that you are running a modern version of docker, and that you are making a recursive clone of this repository and all of its submodules. This assumes that you have jinja installed, as the docker files are generated from the dockerfile.tmpl in the *opt/build/docker* directory.

```
git clone --recursive https://github.com/RedisAI/RedisAI
cd RedisAI/opt/build/docker
make

# note, to build with GPU support instead
make GPU=1
```

After this, you can run the created docker and mount your source code with the following command, from within the RedisAI folder.
Assuming that the docker was built for ubuntu bionic machine with cpu-only support (the default parameters) from master branch, you can run:

```
docker run -v "`pwd`:/build" -it redislabs/redisai:edge-cpu-bionic bash
```

Continue to edit files on your local machine, and rebuild as needed within the docker, by running the command below, from */build* in the docker:

```make -C opt all```

**Building on bare metal**

The instructions below apply to **Ubuntu 18.04 only**. RedisAI can be built on other platforms, as documented is the [system-setup.py file](https://github.com/RedisAI/RedisAI/blob/master/opt/system-setup.py).  This assumes that you're cloning the RedisAI repository.

```
git clone --recursive https://github.com/RedisAI/RedisAI
cd RedisAI
sudo apt-get -qq update -y
sudo apt-get -qq install -y \
    build-essential \
    ca-certificates curl wget unzip \
    gawk \
    libopenblas-dev libmpich-dev \
    git-lfs clang-format-10
```

Ensure that clang-format points to clang-format-10:

```
sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-10 10
```

Install cmake v3.19.5 [from the cmake repository](https://github.com/Kitware/CMake/releases/tag/v3.19.5).

**Building on bare metal, with build scripts**

These instructions apply to **Ubuntu 16.04 and 18.04**. RedisAI can be built on other platforms, but these are the supported Platforms. This assumes you're cloning the RedisAI repository.

```
git clone --recursive https://github.com/RedisAI/RedisAI
cd RedisAI
sudo ./opt/system-setup.py
```

**Building**

To compile RedisAI, run *make -C opt all*, from the root of the repository.

### Testing

**Running Tests**

The module includes a basic set of unit tests and integration tests, split across common and backend specific files. To run them you'll need:

* lcov (for coverage analysis, on Linux)
* Python and a few packages (e.g. pytest, redis, etc.)
* redis-server in your PATH, or in `../redis/src`.

To run all tests in a Python virtualenv, follow these steps:

    $ mkdir -p .env
    $ virtualenv .env
    $ source .env/bin/activate
    $ pip install -r tests/flow/tests_setup/test_requirements.txt
    $ make -C opt test

**Integration tests**

Integration tests are based on RLTest, and specific setup parameters can be provided
to configure tests. By default the tests will be ran for all backends and common commands, and with variation of persistency and replication.

To understand what test options are available simply run:

    $ make -C opt help

For example, to run the tests strictly designed for the TensorFlow backend, follow these steps:

    $ make -C opt test TEST=tests_tensorflow.py

**Coverage**

For coverage analysis we rely on `lcov` that can be run by following these steps:

    $ make -C opt build COV=1 SHOW=1
    $ make -C opt test COV=1 SHOW=1
