# RedisAI Development Backends

This document describes how a backend for RedisAI can be built, from this repository. It highlights the supported compilation devices on a per-backend basis, and highlights the tools and commands required.  Unless indicated otherwise, a backend is compiled in a docker, which is responsible for the configuration and installation of all tools required for a given backend on a per-platform basis.

To follow these instructions, this repository must be cloned with all of its submodules (i.e *git clone --recursive https://github.com/RedisLabsModules/redisai*)

GNU Make is used as a runner for the dockerfile generator. Python is the language used for the generator script, and jinja is the templating library used to create the docker file from a template *dockerfile.tmpl* that can be found in the directory of a given backend listed below.

## Tools

Buiding the backends requires installation of the following tools:

1. gnu make
1. python (3.0 or higher)
1. docker
1. jinja2

On ubuntu bionic these can be installed by running:

* sudo apt install python3 python3-dev make docker
* pip install --user jinja

-------

## Backends

### onnxruntime

**Compilation target devices:**

1. x86\_64 bit linux systems

1. x86\_64 bit linux systems with a GPU

**Directory:** opt/build/onnxruntime

**Build options:**

1. To build run *make*

1. To build with GPU support on x86\_64 run *make GPU=1*

Note: onnxruntime library is built with DISABLE_EXTERNAL_INITIALIZERS=ON build flag. This means that loading ONNX models that use external files to store the initial (usually very large) values of the model's operations, is invalid. That is, initializers values must be part of the serialized model (which is the standard use case)  
