# RedisAI Development Backends

This document describes how ONNXRuntime backend can be built from this repository.
We build the ONNXRuntime library with the DISABLE_EXTERNAL_INITIALIZERS=ON build flag. As a result, loading ONNX models that use external files to store the initial (usually very large) values of the model's operations, is invalid. Hence, initialization values must be part of the serialized model, which is also the standard use case.

It is compiled in a docker, which is responsible for the configuration and installation of all tools required the build process.

To follow these instructions, this repository must be cloned with all of its submodules (i.e *git clone --recursive https://github.com/redisai/redisai*)

GNU Make is used as a runner for the dockerfile generator. Python is the language used for the generator script, and jinja is the templating library used to create the docker file from the template *dockerfile.tmpl*, located in the `/opt/build/onnxruntime` directory.

### Tools

Building the backend requires installation of the following tools:

1. gnu make
1. python (3.0 or higher)
1. docker
1. jinja2

On ubuntu bionic these can be installed by running the following steps, to install python3, create a virtual environment, and install the jinja templating dependency. Replace */path/to/venv* with your desired virtualenv location.

```
sudo apt install python3 python3-dev make docker
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install jinja2
```

-------

**Compilation target devices:**

1. x86\_64 bit linux systems

1. x86\_64 bit linux systems with a GPU

**Directory:** opt/build/onnxruntime

**Build options:**

1. To build run *make*

1. To build with GPU support on x86\_64 run *make GPU=1*
