# RedisAI Development

The following sections discusses topics relevant to the development of the RedisAI module itself. We'll start by referring to the general design, followed by the source code layout, and how to prepare your local test and development environment.

## General Design

RedisAI bundles together best-of-breed technologies for delivering stable and fast model serving. To do so, we need to abstract from what each specific DL/ML framework offers and provide common data structures and APIs to the DL/ML domain. 


As a way of representing tensor data we've embraced [dlpack](https://github.com/dmlc/dlpack) - a community effort to define a common tensor data structure that can be shared by different frameworks, supported by cuPy, cuDF, DGL, TGL, PyTorch, and MxNet.

**Data Structures**

RedisAI provides the following data structures:

* **Tensor**: represents an n-dimensional array of values
* **Model**: represents a frozen graph by one of the supported DL/ML framework backends
* **Script**: represents a [TorchScript](https://pytorch.org/docs/stable/jit.html)

## Source code layout

Inside the root are the following important directories:

* `src`: contains the RedisAI implementation, written in C.
* `opt`: contains the helper scripts to setup the development environment.
* `tests`: contains the unit and flow tests, implemented with python and RLTest.
* `deps`: contains libraries RedisAI uses.

We'll focus mostly on `src`, where the RedisAI implementation is contained,
exploring what there is inside each file. The order in which files are
exposed is the logical one to follow in order to disclose different layers
of complexity incrementally. 

**redisai.c**


This is the entry point of the RedisAI module, responsible for registering the new commands in the Redis server, and containing all command functions to be called. This file is also responsible for exporting of Tensor, Script and Model APIs to other Modules.

**tensor.h**

Contains the helper methods for both creating, populating, managing and destructing the Tensor data structure, and methods to manage parsing and replying of tensor related commands or operations.

**model.h**

Contains the helper methods for both creating, populating, managing and destructing the Model data structure.
Contains also methods to manage parsing and replying of model related commands or operations, that take place in the context of the Redis Main thread.
The helper methods that are related to async background work are available at `model_script_run_session.h` header file.

**script.h**

Contains the helper methods for both creating, populating, managing and destructing the PyTorch Script data structure.
Contains also methods to manage parsing and replying of script related commands or operations, that take place in the context of the Redis Main thread.
The helper methods that are related to async background work are available at `model_script_run_session.h` header file.

**dag.h**

Contains the helper methods for both parsing, running the command in the background, and replying DAG structured commands.

**run_info.h**

Contains the structure and headers to create, initialize, get, reset, and free the structures that represent the context in which RedisAI blocking commands operate, namely `RedisAI_RunInfo` and the newly added `RAI_DagOp`. 

**model_script_run_session.h**

Contains the methods that are related to async background work that 
was triggered by either MODELRUN or SCRIPTRUN Command and Called within `RedisAI_Run_ThreadMain`.
This file also contains the function signatures of the reply callbacks to be called in order to reply to the clients, after the background work on MODELRUN and SCRIPTRUN is done.

**background_workers.h**

Contains the structure to manage the per-device queues, used for decoupling the work from the main thread to the background worker threads. For each of the incoming ModelRun, ScriptRun, and DagRun commands, the request is queued and evaded asynchronously to one the device queues.

**stats.h**

Contains the structure and headers to create, initialize, get, reset, and free run-time statics, like call count, error count, and aggregate durations of ModelRun and ScriptRun sessions.
The statistics are ephemeral, meaning that they are not saved to the key space, and on maximum live for the duration of the DB uptime.

**err.h**

Contains the structure and headers for a formal API to create, initialize, get, reset, and free errors among different backends.

**backends.h and backends directory**

Contains the structure and headers methods required to register a new backend to be loaded by the module. 
 To do so, the backend needs to implement and export the following methods:
 
  * `model_create_with_nodes`:  A callback function pointer that creates a
  model given the RAI_ModelOpts and input and output nodes.
 
  * `model_create`:  A callback function pointer that creates a model given
  the RAI_ModelOpts.
 
  * `model_run`:  A callback function pointer that runs a model given the
  RAI_ModelRunCtx pointer.
 
  * `model_serialize`:  A callback function pointer that serializes a model
  given the RAI_Model pointer.
 
  * `script_create`:  A callback function pointer that creates a script.
 
  * `script_free`:  A callback function pointer that frees a script given
  the RAI_Script pointer.
 
  * `script_run`:  A callback function pointer that runs a model given the
  RAI_ScriptRunCtx pointer.


Within the `backends` folder you will find the implementations code required to support the following DL/ML identifiers and respective backend libraries:

* **TF**: `tensorflow.h` and `tensorflow.c` exporting the functions to to register the TensorFlow backend
* **TFLITE**: `tflite.h` and `tflite.c` exporting the functions to to register the TensorFlow Lite backend
* **TORCH**: `torch.h` and `torch.c` exporting the functions to to register the PyTorch backend
* **ONNX**: `onnxruntime.h` and `onnxruntime.c` exporting the functions to to register the ONNXRuntime backend


## Building and Testing
You can compile and build the module from its source code.

**Prerequisites**

* CUDA needs to be installed for GPU support.
* CMake 3.0 or higher needs to be installed.
* Redis v4.0.9 or greater.

**Get the Source Code**

You can obtain the module's source code by cloning the project's repository using git like so:

    $ git clone https://github.com/RedisAI/RedisAI

Switch to the project's directory with:

    $ cd RedisAI

**Building the Dependencies**

Use the following script to download and build the libraries of the various RedisAI backends (TensorFlow, PyTorch, ONNXRuntime) for your platform with GPU support:

    $ bash get_deps.sh

Alternatively, you can run the following to fetch the CPU-only backends.

    $ bash get_deps.sh cpu

**Building the Module**

Once the dependencies have been built, you can build the RedisAI module with:

    $ make -C opt build
    
**Running Tests**

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
    $ make -C opt test

Integration tests are based on RLTest, and specific setup parameters can be provided
to configure tests.

For example, to run the tests strictly designed for the TensorFlow backend, follow these steps:

    $ make -C opt test TEST=tests_tensorflow.py

**Integration Tests Coverage**

For coverage analysis we rely on lcov, and you can easily run the tests on your local machine following these steps:

    $ make -C opt build COV=1 SHOW=1
    $ make -C opt test COV=1 SHOW=1 AOF=0 SLAVES=0

Setting `AOF=0 SLAVES=0` is not mandatory but will not run AOF and replication tests, thus getting us results faster.
