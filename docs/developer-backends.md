# RedisAI Development Backends

This document describes how a backend for RedisAI can be built, from this repository. It highlights the supported compilation devices on a per-backend basis, and highlights the tools and commands required.  Unless indicated otherwise, a backend is compiled in a docker, which is responsible for the configuration and installation of all tools required for a given backend on a per-platform basis.

To follow these instructions, this repository must be cloned with all of its submodules (i.e *git clone --recursive https://github.com/RedisLabsModules/redisai*)

GNU Make is used as a runner for the dockerfile generator. Python is the language used for the generator script, and jinja is the templating library used to create the docker file from the template.

## Tools

Buiding the backends requires installation of the following tools:

1. gnu make
1. python (3.0 or higher)
1. docker
1. jinja2  jinja is used to generate the platform dockerfile from a *dockerfile.tmpl* that can be found in the directory of a given backend listed below.

On ubuntu bionic these can be installed by running:

* sudo apt install python3 python3-dev make docker
* pip install --user jinja

-------

## Backends

### onnxruntime

**Compilation target devices:**

1. x86\_64 bit linux systems

1. x86\_64 bit linux systems with a GPU

1. jetson devices

**Directory:** opt/build/onnxruntime

**Build options:**

1. To build run *make*

1. To build with GPU support on x86\_64 run *make GPU=1*

1. Should you want to build multiple targets from a shared directory, run *make DOCKER_SUFFIX=<yoursuffix>* on your target system. For example, if building on an arm and x64 workload, from a shared directory run:
    * On x86: make DOCKER\_SUFFIX=x86\_64

    * On arm: make DOCKER\_SUFFIX=arm
