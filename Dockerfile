FROM redis AS builder

ENV DEPS "build-essential git ca-certificates curl unzip wget libgomp1 patchelf"

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
    DEPS_DIRECTORY=deps bash ./get_deps.sh cpu

# Build the source
RUN set -ex;\
    rm -rf build;\
    mkdir -p build;\
    cd build;\
    cmake -DDEPS_PATH=../deps/install ..;\
    make;\
    cd ..

# Package the runner
FROM redis
ENV LD_LIBRARY_PATH /usr/lib/redis/modules/

RUN set -e; apt-get -qq update; apt-get install -y libgomp1

RUN set -ex;\
    mkdir -p "$LD_LIBRARY_PATH";

COPY --from=builder /redisai/build/redisai.so "$LD_LIBRARY_PATH"
COPY --from=builder /redisai/deps/install/lib/*.so* "$LD_LIBRARY_PATH"

WORKDIR /data
EXPOSE 6379
CMD ["--loadmodule", "/usr/lib/redis/modules/redisai.so"]
