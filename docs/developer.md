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

This is the entry point of the RedisAI module, responsible for registering the new commands in the Redis server, and containing all command functions to be called. This file is also responsible for exporting of Tensor, Script and Model APIs to other Modules.

**tensor.h**

Contains the helper methods for both creating, populating, managing and freeing the Tensor data structure, as well as the methods for managing, parsing and replying to tensor related commands or operations.

**model.h**

Contains the helper methods for creating, populating, managing and freeing the Model data structure. Also contains methods for managing, parsing and replying to model related commands or operations, that take place in the context of the Redis main thread.

The helper methods that are related to async background work are available at `model_script_run_session.h` header file.

**script.h**

Contains the helper methods for creating, populating, managing and freeing the PyTorch Script data structure. Also contains methods for managing, parsing and replying to script related commands or operations, that take place in the context of the Redis main thread.

The helper methods that are related to async background work are available at `model_script_run_session.h` header file.

**dag.h**

Contains the helper methods for parsing, running the command in the background, and replying to DAG structured commands.

**run_info.h**

Contains the structure and headers that create, initialize, get, reset, and free the structures that represent the context in which RedisAI blocking commands operate, namely `RedisAI_RunInfo` and the `RAI_DagOp`.

**model_script_run_session.h**

Contains the methods that are related to async background work that was triggered by either `MODELRUN` or `AI.SCRIPTRUN` commands, and called from `RedisAI_Run_ThreadMain`.

This file also contains the function signatures of the reply callbacks to be called in order to reply to the clients, after the background work on `AI.MODELRUN` and `AI.SCRIPTRUN` is done.

**background_workers.h**

Contains the structure for managing per-device queues that are used for decoupling the work from the main thread to the background worker threads. For each of the incoming `AI.MODELRUN`, `AI.SCRIPTRUN`, and `AI.DAGRUN` commands, the request is queued and then executed asynchronously to one the device queues.

**stats.h**

Contains the structure and headers that create, initialize, get, reset, and free run-time statics, like call count, error count, and aggregate durations of `AI.MODELRUN` and `AI.SCRIPTRUN` sessions.

The statistics are ephemeral, meaning that they are not persisted to the Redis key space, and are reset when the server is restarted.

**err.h**

Contains the structure and headers of an API for creating, initializing, getting, resetting, and freeing errors of the different backends.

**backends.h and backends directory**

Contains the structure and headers methods required to register a new backend that can be loaded by the module.

To do so, the backend needs to implement and export the following methods:

* `model_create_with_nodes`: A callback function pointer that creates a
  model given the `RAI_ModelOpts` and input and output nodes.
* `model_create`: A callback function pointer that creates a model given
the `RAI_ModelOpts`.
* `model_run`: A callback function pointer that runs a model given the
`RAI_ModelRunCtx` pointer.
* `model_serialize`: A callback function pointer that serializes a model
given the `RAI_Model` pointer.
* `script_create`: A callback function pointer that creates a script.
* `script_free`: A callback function pointer that frees a script given
the `RAI_Script` pointer.
* `script_run`: A callback function pointer that runs a model given the
`RAI_ScriptRunCtx` pointer.

Within the `backends` folder you will find the implementations code required to support the following DL/ML identifiers and respective backend libraries:

* **TF**: `tensorflow.h` and `tensorflow.c` exporting the functions to to register the TensorFlow backend
* **TFLITE**: `tflite.h` and `tflite.c` exporting the functions to to register the TensorFlow Lite backend
* **TORCH**: `torch.h` and `torch.c` exporting the functions to to register the PyTorch backend
* **ONNX**: `onnxruntime.h` and `onnxruntime.c` exporting the functions to to register the ONNXRuntime backend

## Building and Testing
You can compile and build the module from its source code - refer to the [Building and Running section](quickstart.md#building-and-running) of the Quickstart page for instructions on how to do that.

**Configuring the build environment**

Detailed instructions on configuring your build environment can be found in the [system-setup.py file](https://github.com/RedisAI/RedisAI/blob/master/opt/system-setup.py).  This script will install dependencies for building RedisAI - and is the recommended way of installing build dependencies.

Alternatively, the RedisAI source code can be mounted in a docker, and built there, but edited from the external operating system. As per the example below, try running an Ubuntu 18.04 docker, with the RedisAI source code mounted in /build, run the following command. Note: This should be run from the root of the repository, and it is assumed that docker is installed.

```docker run -v `pwd`:/build -t ubuntu:bionic bash```

On x86 based Linuxes, RedisAI builds on Ubuntu 16.04, Ubuntu 18.04, and Debian buster. On Arm-based Linuxes, the Nvidia dockers from the [nvidia cuda project](https://hub.docker.com/r/nvidia/cuda) are the baseline.

Currently, the following dependencies are required:

* [CMake > 3.15](https://github.com/Kitware/CMake/releases)
* GCC > 6.x , but < 10
* unzip and patch - These are needed by get_deps as documented in the [quickstart](quickstart.md#building-and-running)
* [NetworkX](https://networkx.org/)
* [numpy](https://pypi.org/project/numpy/)
* [OpenBLAS](https://www.openblas.net/)

If you intend on running memory tests, please install valgrind.

**Running Tests**

The module includes a basic set of unit tests and integration tests, split across common and backend specific files. To run them you'll need:

* lcov (for coverage analysis, on Linux)
* Python and a few packages (e.g. pytest, redis, etc.)
* redis-server in your PATH, or in `../redis/src`.

To run all tests in a Python virtualenv, follow these steps:

    $ mkdir -p .env
    $ virtualenv .env
    $ source .env/bin/activate
    $ pip install -r tests/flow/test_requirements.txt
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
