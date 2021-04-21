# tensorflowlite (tflite)

### arm64

The arm build of tflite currently occurs on **jetson arm devices** only, as portions of the root filesystem of the Jetson device are mounted during the build. Given the symlinks that exist on the device between things in /usr/lib to /etc/alternatives, and in turn to /usr/local/cuda, which is itself a symlink to /usr/local/cuda-10.2, this is the current philosophy.

WThe *build_arm* target in the [Makefile](Makefile) describes the process in detail. The code to build the base docker build image can be found in the [automata repository](https://github.com/RedisLabsModules/automata/tree/master/dockers/buildsystem/bazelbuilder). The *bazelbuilder* image is published to the [redisfab dockerhub repositories](https://hub.docker.com/r/redisfab/).