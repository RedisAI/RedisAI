# RedisAI dependency builds

Platform dependency build systems are located in this folder. Dependencies are to be pre-built, and published to S3. To do so, they rely *(ultimately)* on running **make build publish** in a given directory. The goal is for this to be true on all target platforms (x86_64, arm64), though at this time it's only true for: tensorflowlite.

## Background

Items are built in docker images, for the target platform whenever possible. If needed (i.e a future planned MacOS build) items are built on the dedicated hardware. There are design wrinkles to each build. Though the ideal is to build a base docker (see the [automata repository](https://github.com/redislabsmodules/automata). That base docker is then used as the base build system injector for the dependency itself.  A docker image is built from the base docker, accepting externalized variables such as the dependency version. Compilation of external requirements takes place in a build file, mounted inside the docker image.

Ideally a per-platform Docker file (i.e Dockerfile.x64, Dockerfile.arm) will exist in the underlying folder, assuming building within a docker is tenable.