FROM ubuntu

ENV DEPS "build-essential git ca-certificates"

# Set up a build environment
RUN set -ex;\
    deps="$DEPS";\
    apt-get update;\
    apt-get install -y --no-install-recommends $deps;

# Build the source
ADD ./ /redisai

WORKDIR /redisai
RUN set -ex;\
    bash ./get_deps.sh;\
    make install;

# Package the runner
FROM redis
ENV LD_LIBRARY_PATH /usr/lib/redis/modules

WORKDIR /data
RUN set -ex;\
    mkdir -p "$LD_LIBRARY_PATH";

COPY --from=builder /redisai/install/redisai.so "$LD_LIBRARY_PATH"
COPY --from=builder /redisai/deps/libtensorflow/lib/libtensorflow.so "$LD_LIBRARY_PATH"
COPY --from=builder /redisai/deps/libtensorflow/lib/libtensorflow_framework.so "$LD_LIBRARY_PATH"

RUN echo $LD_LIBRARY_PATH

EXPOSE 6379
CMD ["redis-server", "--loadmodule", "/usr/lib/redis/modules/redisai.so"]
