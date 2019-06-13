FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 AS builder

ENV DEPS "build-essential git ca-certificates curl unzip wget libgomp1 patchelf"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt update && apt -y upgrade
RUN apt-get install -y libgomp1
RUN apt-get install -y wget

RUN wget http://mirrors.kernel.org/ubuntu/pool/universe/j/jemalloc/libjemalloc2_5.1.0-1_amd64.deb
RUN dpkg -i libjemalloc2_5.1.0-1_amd64.deb
RUN wget http://mirrors.kernel.org/ubuntu/pool/universe/r/redis/redis-tools_4.0.11-2_amd64.deb
RUN dpkg -i redis-tools_4.0.11-2_amd64.deb
RUN wget http://mirrors.kernel.org/ubuntu/pool/universe/r/redis/redis-server_4.0.11-2_amd64.deb
RUN dpkg -i redis-server_4.0.11-2_amd64.deb

# install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.4-Linux-x86_64.sh /cmake-3.12.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Set up a build environment
RUN set -ex;\
    deps="$DEPS";\
    apt-get update;\
    apt-get install -y --no-install-recommends $deps

# Get the dependencies
WORKDIR /redisai
ADD ./ /redisai
RUN set -ex;\
    mkdir -p deps;\
    DEPS_DIRECTORY=deps bash ./get_deps.sh

# Build the source
RUN set -ex;\
    rm -rf build;\
    mkdir -p build;\
    cd build;\
    cmake -DDEPS_PATH=../deps/install ..;\
    make;\
    cd ..

# Package the runner
FROM builder
ENV LD_LIBRARY_PATH /usr/lib/redis/modules/

RUN set -ex;\
    mkdir -p "$LD_LIBRARY_PATH";

COPY --from=builder /redisai/build/redisai.so "$LD_LIBRARY_PATH"
COPY --from=builder /redisai/deps/install/lib/*.so* "$LD_LIBRARY_PATH"

WORKDIR /data
EXPOSE 6379
CMD ["redis-server", "--loadmodule", "/usr/lib/redis/modules/redisai.so"]
